import os, json, asyncio, re
import httpx
import chainlit as cl
from chainlit.input_widget import Slider, Switch
from dotenv import load_dotenv

load_dotenv()

RUNPOD_BASE = os.getenv("RUNPOD_BASE", "https://api.runpod.ai/v2")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")
TIMEOUT = float(os.getenv("CLIENT_TIMEOUT_S", "90"))
URL_RUNSYNC = f"{RUNPOD_BASE}/{ENDPOINT_ID}/runsync"

DEFAULT_SAMPLING = {
    "max_tokens": 200,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "stop": ["<|im_end|>"]
}

DEFAULT_FLAGS = {
    "enable_validation": True,
    "micro_retry": True,
    "use_llm_router": True
}

ROUTER_LABELS = [
    "resume_guidance",
    "job_description",
    "job_resume_match",
    "recruiting_strategy",
    "ats_keywords",
    "small_talk",
    "general_qna"
]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Seeded domains (start with "1. ")
SEEDED = {"resume_guidance", "recruiting_strategy", "ats_keywords"}

# 1-shot examples
RESUME_EXAMPLE = (
    "1. Led migration of 50+ services to Kubernetes, reducing deployments by 60%\n"
    "2. Built CI/CD with Jenkins + ArgoCD, enabling 20 daily deploys\n"
    "3. Optimized AWS costs by 35% via auto-scaling and rightsizing\n"
    "4. Designed REST APIs handling 10k+ req/sec with FastAPI\n"
    "5. Implemented monitoring with Prometheus/Grafana and on-call rota\n"
    "6. Mentored 3 junior engineers in code reviews and design\n"
    "ATS Keywords: Kubernetes, Docker, Jenkins, ArgoCD, Terraform, AWS, FastAPI, Python, CI/CD, Prometheus"
)

JD_EXAMPLE = (
    "Summary: Experienced backend engineer to build scalable APIs.\n"
    "Responsibilities:\n1. Design REST APIs\n2. Optimize DB queries\n3. Implement security\n4. Collaborate with DevOps\n5. Code reviews\n6. Monitor SLIs/SLOs\n"
    "Requirements:\n1. 5+ yrs backend\n2. Python/Go/Node\n3. SQL/NoSQL\n4. AWS/GCP\n5. System design\n6. Communication\n"
)

# Validators
def validate_resume(text: str) -> tuple[bool, list[str]]:
    issues = []
    if len(text.split()) < 50:
        issues.append("len")
    if not re.search(r'^\s*[\d\-\*\•]', text, re.M):
        issues.append("bullets")
    action_verbs = ["led", "built", "developed", "managed", "created", "improved", "optimized", "designed", "implemented", "achieved"]
    if not any(v in text.lower() for v in action_verbs):
        issues.append("verbs")
    if ("ats keywords" not in text.lower()) and text.count(",") < 5:
        issues.append("ats_line")
    return (len(issues) == 0), issues

def validate_jd(text: str) -> tuple[bool, list[str]]:
    issues = []
    if len(text.split()) < 100:
        issues.append("len")
    if sum(s in text.lower() for s in ["responsibilities", "requirements"]) < 2:
        issues.append("sections")
    if len(re.findall(r'^\s*[\d\-\*\•]', text, re.M)) < 8:
        issues.append("bullets")
    return (len(issues) == 0), issues

def validate_match(text: str) -> tuple[bool, list[str]]:
    issues = []
    if len(text.split()) < 60:
        issues.append("len")
    if not re.search(r'\b\d{1,3}\s*%|\bscore[:\s]+\d{1,3}', text, re.I):
        issues.append("score")
    if sum(w in text.lower() for w in ["match", "gap", "skill"]) < 2:
        issues.append("content")
    # Check for rationale keywords
    if not any(w in text.lower() for w in ["because", "due to", "since", "as", "reason", "strong", "weak", "partial"]):
        issues.append("rationale")
    return (len(issues) == 0), issues

def validate_recruit(text: str) -> tuple[bool, list[str]]:
    issues = []
    if len(text.split()) < 60:
        issues.append("len")
    channels = ["linkedin", "referral", "meetup", "university", "conference", "github", "stackoverflow"]
    if sum(c in text.lower() for c in channels) < 3:
        issues.append("channels")
    if not any(w in text.lower() for w in ["week", "month", "quarter", "weekly", "monthly", "quarterly"]):
        issues.append("cadence")
    return (len(issues) == 0), issues

def validate_ats(text: str) -> tuple[bool, list[str]]:
    k = max(text.count(",") + 1, len(re.findall(r'^\s*[\d\-\*\•]', text, re.M)))
    return (k >= 15), ([] if k >= 15 else ["count"])

VALIDATORS = {
    "resume_guidance": validate_resume,
    "job_description": validate_jd,
    "job_resume_match": validate_match,
    "recruiting_strategy": validate_recruit,
    "ats_keywords": validate_ats
}

async def classify_with_llm(user_prompt: str) -> tuple[str | None, float]:
    """Classify user prompt using LLM router with JSON response."""
    router_prompt = f"""<|im_start|>system
You are a router that classifies HR/career requests into one of:

- resume_guidance
- job_description
- job_resume_match
- recruiting_strategy
- ats_keywords
- small_talk
- general_qna

Key rules:

- "strategy to hire/recruit/get/find/attract/source developers/candidates" → recruiting_strategy
- "hiring pipeline / talent acquisition / onboarding plan" → recruiting_strategy
- "job description / JD / job posting / job ad" → job_description
- "match score / score: / alignment / fit score" → job_resume_match
- "ATS / applicant tracking / keywords" → ats_keywords
- Greetings ("hello", "how are you", "hey") → small_talk

Return strict JSON: {{"intent":"...", "confidence":0.0-1.0}}. No extra text.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    sampling = {
        "max_tokens": 60,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "stop": ["<|im_end|>"]
    }

    try:
        response = await call_endpoint(router_prompt, sampling)
        text = extract_text(response) or ""

        # Parse JSON robustly
        match = re.search(r'\{.*?\}', text, flags=re.S)
        if not match:
            return None, 0.0

        data = json.loads(match.group(0))
        intent = data.get("intent")
        confidence = float(data.get("confidence") or 0.0)

        # Validate intent is in allowed labels
        if intent in ROUTER_LABELS:
            return intent, confidence

        return None, 0.0
    except Exception:
        # Fallback to heuristic if router fails
        return None, 0.0

# ChatML builder
def build_chatml(domain: str, user_prompt: str) -> str:
    if domain == "resume_guidance":
        sys_txt = (
            "You are a career coach specializing in resumes. Provide 6-8 numbered bullets using action verbs. "
            'End with: "ATS Keywords: ..." (comma-separated).\n\nExample:\n' + RESUME_EXAMPLE
        )
    elif domain == "job_description":
        sys_txt = (
            "You are an HR specialist. Create a JD with sections: Summary (2-3 lines), Responsibilities (6-8 bullets), "
            "Requirements (6-8 bullets). Keep 150-200 words.\n\nExample:\n" + JD_EXAMPLE
        )
    elif domain == "job_resume_match":
        sys_txt = (
            "You are a technical recruiter. Provide: Score (0-100) with brief explanation of why this score, "
            "Matches (5+ skills/experiences aligned), Gaps (3+ missing skills), Next steps (3+ recommendations). "
            "Explain your reasoning for the score. Keep 100-150 words."
        )
    elif domain == "recruiting_strategy":
        sys_txt = (
            "You are a recruiting strategist. Provide 4–6 numbered steps. Include at least 3 of these exact terms: "
            "LinkedIn, referral, meetup, university, conference, GitHub, Stack Overflow. Include cadence "
            "(weekly/monthly/quarterly) and 1–2 metrics (response rate %, time-to-hire). Keep 100–150 words."
        )
    elif domain == "ats_keywords":
        sys_txt = (
            "You are a resume optimization specialist. Provide 20–40 ATS keywords comma-separated (80–120 words). "
            "Include technical tools, domains, certifications, and soft skills."
        )
    elif domain == "small_talk":
        sys_txt = "You are a friendly assistant. Respond warmly in 1–3 sentences."
        return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:  # general_qna
        sys_txt = "You are a helpful assistant. Provide concise, accurate answers."
        return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Unreachable, but keeping for clarity of HR domains
    seed = "1. " if domain in SEEDED else ""
    return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{seed}"

# HR Task Templates
HR_TASKS = {
    "resume_guidance": {
        "label": "📄 Resume Guidance",
        "prompt": "Optimize my resume for Senior Backend Engineer (Python, FastAPI, AWS).",
        "domain": "resume_guidance"
    },
    "job_description": {
        "label": "📝 Job Description Help",
        "prompt": "Write a job description for Senior DevOps Engineer at a cloud infrastructure company.",
        "domain": "job_description"
    },
    "ats_keywords": {
        "label": "🔍 ATS Keywords",
        "prompt": "List ATS keywords for DevOps Engineer (Kubernetes, Terraform, CI/CD).",
        "domain": "ats_keywords"
    },
    "job_resume_match": {
        "label": "🎯 Job-Resume Match",
        "prompt": "Score match: candidate has Docker, Jenkins, Python vs JD requiring AWS, Kubernetes, Terraform, CI/CD.",
        "domain": "job_resume_match"
    },
    "recruiting_strategy": {
        "label": "📊 Recruiting Strategy",
        "prompt": "Sourcing strategy to hire 5 senior ML engineers in 3 months.",
        "domain": "recruiting_strategy"
    }
}

def extract_text(payload: dict) -> str:
    """Extract text from vLLM response (supports both raw vLLM and handler endpoint)."""
    out = payload.get("output", [])

    # Handler endpoint format: dict with choices[0].text
    if isinstance(out, dict):
        choices = out.get("choices", [])
        if choices:
            return choices[0].get("text", "") or (choices[0].get("tokens", [""])[0] if "tokens" in choices[0] else "")

    # Raw vLLM format: list with choices[0].tokens
    if isinstance(out, list) and out:
        choices = out[0].get("choices", [])
        if choices:
            # Try tokens first (raw vLLM), then text (handler)
            if "tokens" in choices[0]:
                tokens = choices[0].get("tokens", [])
                return tokens[0] if tokens else ""
            return choices[0].get("text", "")

    return ""

def extract_usage(payload: dict) -> dict | None:
    """Extract usage stats from response (supports both formats)."""
    out = payload.get("output", [])

    # Handler endpoint format: dict with usage
    if isinstance(out, dict):
        return out.get("usage")

    # Raw vLLM format: list with usage
    if isinstance(out, list) and out:
        return out[0].get("usage")

    return None

async def call_endpoint(prompt: str, sampling: dict) -> dict:
    """Call RunPod endpoint with ChatML prompt."""
    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": sampling
        }
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(URL_RUNSYNC, headers=HEADERS, json=payload)
        if r.status_code != 200:
            raise Exception(f"Error {r.status_code}: {r.text[:300]}")
        return r.json()

async def generate_with_validation(domain: str, user_prompt: str, sampling: dict, enable_validation: bool, micro_retry: bool) -> dict:
    """Generate response with optional validation and micro-retry."""
    # First attempt
    chat = build_chatml(domain, user_prompt)
    response = await call_endpoint(chat, sampling)
    text = extract_text(response)
    usage = extract_usage(response)
    exec_time = response.get("executionTime", 0)

    # Validate if enabled
    if enable_validation and domain in VALIDATORS:
        ok, issues = VALIDATORS[domain](text)

        # Micro-retry if validation fails and micro_retry enabled
        if not ok and micro_retry:
            clarifier = get_clarifier(domain)
            chat_retry = build_chatml(domain, user_prompt + clarifier)
            response2 = await call_endpoint(chat_retry, sampling)
            text2 = extract_text(response2)
            ok2, issues2 = VALIDATORS[domain](text2)

            return {
                "text": text2,
                "ok": ok2,
                "issues": issues2,
                "exec_time": response2.get("executionTime", 0),
                "retried": True,
                "usage": extract_usage(response2)
            }

        return {
            "text": text,
            "ok": ok,
            "issues": issues,
            "exec_time": exec_time,
            "retried": False,
            "usage": usage
        }

    # No validation
    return {
        "text": text,
        "ok": True,
        "issues": [],
        "exec_time": exec_time,
        "retried": False,
        "usage": usage
    }

def get_clarifier(domain: str) -> str:
    """Get clarifier text for micro-retry."""
    clarifiers = {
        "resume_guidance": " Ensure 6–8 bullets with action verbs and end with: ATS Keywords: ...",
        "job_description": " Ensure sections: Summary + Responsibilities (6–8) + Requirements (6–8).",
        "job_resume_match": " Ensure Score (0–100) with explanation of why, Matches (5+), Gaps (3+), Next steps (3+). Explain the reasoning behind the score.",
        "recruiting_strategy": " Mention LinkedIn, referral, meetup, university, conference, GitHub, Stack Overflow and cadence weekly/monthly/quarterly.",
        "ats_keywords": " Provide 20–40 comma-separated keywords (80–120 words)."
    }
    return clarifiers.get(domain, "")

@cl.on_chat_start
async def start():
    # Initialize settings in user session
    cl.user_session.set("settings", {
        "temperature": DEFAULT_SAMPLING["temperature"],
        "top_p": DEFAULT_SAMPLING["top_p"],
        "max_tokens": DEFAULT_SAMPLING["max_tokens"],
        "enable_validation": DEFAULT_FLAGS["enable_validation"],
        "micro_retry": DEFAULT_FLAGS["micro_retry"],
        "use_llm_router": DEFAULT_FLAGS["use_llm_router"]
    })

    # Create settings panel
    settings = await cl.ChatSettings([
        Slider(
            id="temperature",
            label="Temperature",
            initial=DEFAULT_SAMPLING["temperature"],
            min=0.0,
            max=1.0,
            step=0.1,
            description="Controls randomness. Lower = more focused, Higher = more creative"
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=DEFAULT_SAMPLING["top_p"],
            min=0.0,
            max=1.0,
            step=0.05,
            description="Nucleus sampling. Controls diversity of responses"
        ),
        Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=DEFAULT_SAMPLING["max_tokens"],
            min=50,
            max=300,
            step=10,
            description="Maximum length of response"
        ),
        Switch(
            id="use_llm_router",
            label="LLM Router",
            initial=DEFAULT_FLAGS["use_llm_router"],
            description="Use LLM to detect intent (more accurate, +200-400ms latency)"
        ),
        Switch(
            id="enable_validation",
            label="Enable Validation",
            initial=DEFAULT_FLAGS["enable_validation"],
            description="Validate outputs against domain-specific quality checks"
        ),
        Switch(
            id="micro_retry",
            label="Micro-Retry",
            initial=DEFAULT_FLAGS["micro_retry"],
            description="Automatically retry with clarifications if validation fails"
        )
    ]).send()

    # Create HR task action buttons
    actions = [
        cl.Action(name=task_id, payload={"prompt": task_data["prompt"], "domain": task_data["domain"]}, label=task_data["label"])
        for task_id, task_data in HR_TASKS.items()
    ]

    # Send welcome message with actions
    await cl.Message(
        content="👋 **Welcome to EvalMatch Career Copilot!**\n\n"
                "Your AI-powered assistant for smarter hiring and career growth.\n\n"
                "**I can help you with:**\n"
                "• 📄 Resume optimization and career guidance\n"
                "• 📝 Job descriptions and postings\n"
                "• 🎯 Resume-job matching and scoring\n"
                "• 🔍 ATS keywords and optimization\n"
                "• 📊 Recruiting strategies and pipelines\n\n"
                "💬 Just ask naturally - I'll understand your intent automatically!\n\n"
                "Or use quick actions below:",
        actions=actions
    ).send()

@cl.on_settings_update
async def settings_update(settings):
    """Handle settings changes."""
    cl.user_session.set("settings", settings)
    await cl.Message(content=f"✅ Settings updated: temp={settings['temperature']}, top_p={settings['top_p']}, "
                              f"llm_router={settings['use_llm_router']}, validation={settings['enable_validation']}, "
                              f"micro_retry={settings['micro_retry']}").send()

@cl.action_callback("resume_guidance")
async def on_resume_guidance(action: cl.Action):
    """Handle Resume Guidance action."""
    await process_hr_task(action.payload["prompt"], action.payload["domain"])

@cl.action_callback("job_description")
async def on_job_description(action: cl.Action):
    """Handle Job Description action."""
    await process_hr_task(action.payload["prompt"], action.payload["domain"])

@cl.action_callback("ats_keywords")
async def on_ats_keywords(action: cl.Action):
    """Handle ATS Keywords action."""
    await process_hr_task(action.payload["prompt"], action.payload["domain"])

@cl.action_callback("job_resume_match")
async def on_job_resume_match(action: cl.Action):
    """Handle Job-Resume Match action."""
    await process_hr_task(action.payload["prompt"], action.payload["domain"])

@cl.action_callback("recruiting_strategy")
async def on_recruiting_strategy(action: cl.Action):
    """Handle Recruiting Strategy action."""
    await process_hr_task(action.payload["prompt"], action.payload["domain"])

async def process_hr_task(prompt: str, domain: str):
    """Process HR task with current settings."""
    settings = cl.user_session.get("settings")

    sampling = {
        "max_tokens": settings.get("max_tokens", DEFAULT_SAMPLING["max_tokens"]),
        "temperature": settings.get("temperature", DEFAULT_SAMPLING["temperature"]),
        "top_p": settings.get("top_p", DEFAULT_SAMPLING["top_p"]),
        "repetition_penalty": DEFAULT_SAMPLING["repetition_penalty"],
        "stop": DEFAULT_SAMPLING["stop"]
    }

    enable_validation = settings.get("enable_validation", DEFAULT_FLAGS["enable_validation"])
    micro_retry = settings.get("micro_retry", DEFAULT_FLAGS["micro_retry"])

    # Action buttons bypass router (domain is pre-determined)
    router_info = {"method": "action_button", "confidence": 1.0}

    await send_query(prompt, domain, sampling, enable_validation, micro_retry, router_info)

async def send_query(prompt: str, domain: str, sampling: dict, enable_validation: bool, micro_retry: bool, router_info: dict = None):
    """Core function to send query to RunPod and stream response."""
    if not ENDPOINT_ID or not API_KEY:
        await cl.Message(content="❌ Server missing RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY.").send()
        return

    thinking = cl.Message(content="🤔 Generating response...")
    await thinking.send()

    # Skip validation for non-HR domains
    is_hr_domain = domain in ["resume_guidance", "job_description", "job_resume_match", "recruiting_strategy", "ats_keywords"]
    actual_validation = enable_validation and is_hr_domain
    actual_retry = micro_retry and is_hr_domain

    try:
        result = await generate_with_validation(domain, prompt, sampling, actual_validation, actual_retry)
    except Exception as e:
        await thinking.update(content=f"❌ Request error: {e}")
        return

    # Remove thinking message
    await thinking.remove()

    # Build status message
    status_parts = []

    # Add domain detection info
    if router_info:
        conf = router_info.get("confidence", 0)
        method = router_info.get("method", "heuristic")
        if method == "llm":
            status_parts.append(f"🎯 Detected: {domain} ({conf:.0%})")
        else:
            status_parts.append(f"🎯 Detected: {domain}")

    if result.get("retried"):
        status_parts.append("🔄 Retried")
    if not result.get("ok") and is_hr_domain:
        status_parts.append(f"⚠️ Issues: {', '.join(result.get('issues', []))}")
    if result.get("ok") and is_hr_domain:
        status_parts.append("✅ Validated")

    status = f"\n\n*{' • '.join(status_parts)}*" if status_parts else ""

    # Stream the response
    text = result.get("text", "")
    out = cl.Message(content="")
    await out.send()

    # Stream by words
    for token in text.split():
        await asyncio.sleep(0)  # yield
        await out.stream_token(token + " ")

    # Add status footer
    await out.stream_token(status)
    await out.update()

def detect_domain(prompt: str) -> str | None:
    """Detect HR domain from user prompt (heuristic fallback)."""
    p = prompt.lower().strip()

    # Small talk (greetings)
    if any(w in p for w in ["hello", "hi ", "hey", "how are you", "good morning", "good evening"]):
        return "small_talk"

    # ATS keywords (most specific first)
    if any(w in p for w in ["keyword", "ats", "applicant tracking"]):
        return "ats_keywords"

    # Job-Resume match
    if any(w in p for w in ["match score", "score:", "alignment", "fit score"]):
        return "job_resume_match"

    # Job description
    if any(w in p for w in ["job description", "jd", "job posting", "job ad"]):
        return "job_description"

    # Recruiting strategy (before resume to catch "pipeline", "hire", etc.)
    if any(w in p for w in ["recruit", "sourcing", "hiring", "pipeline", "talent acquisition", "candidate", "hire", "onboard", "attract", "find developers", "get developers", "strategy to"]):
        return "recruiting_strategy"

    # Resume guidance
    if any(w in p for w in ["resume", "cv", "bullet", "achievement", "experience"]):
        return "resume_guidance"

    # If no domain detected, return None
    return None

@cl.on_message
async def on_message(msg: cl.Message):
    """Handle user messages."""
    prompt = msg.content.strip()

    # Get current settings from session
    settings = cl.user_session.get("settings") or {
        "temperature": DEFAULT_SAMPLING["temperature"],
        "top_p": DEFAULT_SAMPLING["top_p"],
        "max_tokens": DEFAULT_SAMPLING["max_tokens"],
        "enable_validation": DEFAULT_FLAGS["enable_validation"],
        "micro_retry": DEFAULT_FLAGS["micro_retry"],
        "use_llm_router": DEFAULT_FLAGS["use_llm_router"]
    }

    sampling = {
        "max_tokens": settings.get("max_tokens", DEFAULT_SAMPLING["max_tokens"]),
        "temperature": settings.get("temperature", DEFAULT_SAMPLING["temperature"]),
        "top_p": settings.get("top_p", DEFAULT_SAMPLING["top_p"]),
        "repetition_penalty": DEFAULT_SAMPLING["repetition_penalty"],
        "stop": DEFAULT_SAMPLING["stop"]
    }

    # Inline override: temp=0.2 in message
    if "temp=" in prompt:
        try:
            val = float(prompt.split("temp=")[1].split()[0])
            sampling["temperature"] = max(0.0, min(1.0, val))
        except Exception:
            pass

    enable_validation = settings.get("enable_validation", DEFAULT_FLAGS["enable_validation"])
    micro_retry = settings.get("micro_retry", DEFAULT_FLAGS["micro_retry"])
    use_llm_router = settings.get("use_llm_router", DEFAULT_FLAGS["use_llm_router"])

    # Domain detection with LLM router
    domain = None
    router_info = {}

    if use_llm_router:
        intent, confidence = await classify_with_llm(prompt)

        if intent and confidence >= 0.6:
            # High confidence - use LLM classification
            domain = intent
            router_info = {"method": "llm", "confidence": confidence}
        elif intent and confidence >= 0.4:
            # Medium confidence - try heuristic, fallback to LLM
            heuristic_domain = detect_domain(prompt)
            domain = heuristic_domain if heuristic_domain else intent
            router_info = {"method": "hybrid", "confidence": confidence}
        else:
            # Low confidence - use heuristic, fallback to general_qna
            domain = detect_domain(prompt) or "general_qna"
            router_info = {"method": "heuristic", "confidence": confidence}
    else:
        # LLM router disabled - use heuristic only
        domain = detect_domain(prompt) or "general_qna"
        router_info = {"method": "heuristic", "confidence": 1.0}

    await send_query(prompt, domain, sampling, enable_validation, micro_retry, router_info)
