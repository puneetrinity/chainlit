import os, json, asyncio, re
import logging
import httpx
import chainlit as cl
from chainlit.input_widget import Slider, Switch
from dotenv import load_dotenv

load_dotenv()

# Setup logger for debug output (add handler only if none exists to avoid clobbering embedded host logging)
logger = logging.getLogger("evalmatch")
logger.setLevel(logging.DEBUG if os.getenv("DEBUG_VARIANTS", "false").lower() == "true" else logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False  # Prevent double emission if parent handlers exist

RUNPOD_BASE = os.getenv("RUNPOD_BASE", "https://api.runpod.ai/v2")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")
TIMEOUT = float(os.getenv("CLIENT_TIMEOUT_S", "90"))
# Retry behavior for transient timeouts
CLIENT_RETRY_ATTEMPTS = int(os.getenv("CLIENT_RETRY_ATTEMPTS", "1"))  # number of retries on timeout
CLIENT_RETRY_BACKOFF_S = float(os.getenv("CLIENT_RETRY_BACKOFF_S", "0.75"))  # base backoff seconds
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

# Global style rules for all HR domains
HR_STYLE_RULES = "No preambles/disclaimers; return only the requested sections; concise, numbered where applicable; no markdown metadata."

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

# Template Registry with Variants
TEMPLATES = {
    "resume_guidance": {
        "seed": True,
        "base": "You are a career coach specializing in resumes. Provide 6-8 numbered bullets using action verbs. End with: \"ATS Keywords: ...\" (comma-separated).",
        "example": RESUME_EXAMPLE,
        "variants": [
            {
                "name": "senior_lead",
                "hint": "Focus on leadership scope (team size, budget, cross-functional impact), strategic initiatives, and measurable business outcomes. Emphasize ownership and mentorship."
            },
            {
                "name": "technical_depth",
                "hint": "Highlight technical depth: specific technologies, architecture decisions, performance optimizations, and technical challenges overcome."
            }
        ]
    },
    "job_description": {
        "seed": False,
        "base": "You are an HR specialist. Create a JD with sections: Summary (2-3 lines), Responsibilities (6-8 bullets), Requirements (6-8 bullets). Keep 150-200 words.",
        "example": JD_EXAMPLE,
        "max_variants": 2,  # Allow both tech and remote variants
        "variants": [
            {
                "name": "tech_deep_dive",
                "hint": "Include specific tech stack details, infrastructure requirements, and compliance/security standards (e.g., PCI, SOC2, HIPAA, ISO 27001) if relevant."
            },
            {
                "name": "remote_distributed",
                "hint": "Emphasize remote work expectations, communication practices, timezone considerations, and distributed team collaboration."
            }
        ]
    },
    "job_resume_match": {
        "seed": False,
        "base": "You are a technical recruiter. Provide: Score (0-100) with brief explanation of why this score, Matches (5+ skills/experiences aligned), Gaps (3+ missing skills), Next steps (3+ recommendations). Explain your reasoning for the score. Keep 100-150 words.",
        "example": None,
        "sampling": {"min_tokens": 260},  # Ensure sufficient length for detailed analysis
        "variants": [
            {
                "name": "quick_screen",
                "hint": "Provide a concise screening assessment (80-120 words). Focus on critical must-haves and deal-breakers."
            }
        ]
    },
    "recruiting_strategy": {
        "seed": True,
        "base": "You are a recruiting strategist. Provide 4–6 numbered steps. Include at least 3 of these exact terms: LinkedIn, referral, meetup, university, conference, GitHub, Stack Overflow. Include cadence (weekly/monthly/quarterly) and 1–2 metrics (response rate %, time-to-hire). Keep 100–150 words.",
        "example": None,
        "variants": [
            {
                "name": "streaming_aware",
                "hint": (
                    "For Java + streaming-like profiles (even if Kafka/Pulsar not stated): "
                    "1) Include Boolean sourcing patterns using adjacent signals: 'event-driven', 'real-time', 'message queue', 'JMS', 'ActiveMQ', 'RabbitMQ', 'Kinesis', 'Pulsar', 'Pub/Sub', 'CDC/Debezium', 'Spring Cloud Stream', 'Project Reactor', 'Akka', 'Netty', 'backpressure', 'low-latency', 'throughput'. "
                    "2) List 4–6 channels with cadence: LinkedIn Recruiter, GitHub code search, Stack Overflow tags, Kafka/Reactive meetups, Java user groups, conference speakers (QCon/Kafka Summit). Include weekly/monthly/quarterly rhythm. "
                    "3) Outreach: personalize by mapping their event-driven/JMS/Reactive signals to Kafka/Pulsar; add a short sample opener. "
                    "4) Screening prompts: backpressure/idempotent consumers; batch→near-real-time migration pitfalls. "
                    "5) Metrics: response rate %, pass-through %, time-to-hire."
                )
            },
            {
                "name": "senior_executive",
                "hint": "For senior/executive roles: focus on executive recruiters, industry connections, board networks, and longer hiring cycles (2-4 months)."
            }
        ]
    },
    "ats_keywords": {
        "seed": False,
        "base": "You are a resume optimization specialist. Provide 20–40 ATS keywords comma-separated (80–120 words). Include technical tools, domains, certifications, and soft skills.",
        "example": None,
        "sampling": {"repetition_penalty": 1.05},  # Slight penalty to reduce keyword repetition
        "variants": [
            {
                "name": "role_specific",
                "hint": "Tailor keywords specifically for {role}: include role-specific tools, frameworks, methodologies, and industry-standard certifications."
            }
        ]
    }
}

def extract_slots(prompt: str) -> dict[str, str | None]:
    """Extract role, stack, and location slots from user prompt."""
    p = prompt.lower()
    slots = {"role": None, "stack": None, "location": None}

    # Role detection
    roles = {
        "android": "Android Developer",
        "ios": "iOS Developer",
        "data engineer": "Data Engineer",
        "data scientist": "Data Scientist",
        "ml engineer": "ML Engineer",
        "machine learning": "ML Engineer",
        "sre": "SRE",
        "site reliability": "SRE",
        "devops": "DevOps Engineer",
        "backend": "Backend Engineer",
        "frontend": "Frontend Engineer",
        "fullstack": "Fullstack Engineer",
        "full stack": "Fullstack Engineer",
        "qa": "QA Engineer",
        "security": "Security Engineer"
    }

    for keyword, role in roles.items():
        if keyword in p:
            slots["role"] = role
            break

    # Stack detection (golang in list, "go" handled separately with word boundaries)
    stacks_exact = ["java", "python", "javascript", "typescript", "golang", "rust", "c++",
                    "kafka", "kubernetes", "k8s", "docker", "react", "angular", "vue",
                    "aws", "gcp", "azure", "terraform", "fastapi", "django", "flask"]

    found_stacks = []
    for s in stacks_exact:
        if s in p:
            found_stacks.append(s)

    # Check for "go" separately with word boundaries to avoid "going", "goal", "cargo", etc.
    if re.search(r'\bgo\b', p):
        found_stacks.append("go")

    if found_stacks:
        slots["stack"] = ", ".join(found_stacks[:3])  # Top 3

    # Location detection (use word boundaries for short names)
    if "remote" in p or "distributed" in p:
        slots["location"] = "Remote"
    elif "san francisco" in p or re.search(r'\bsf\b', p):
        slots["location"] = "San Francisco"
    elif "new york" in p or re.search(r'\bnyc\b', p):
        slots["location"] = "New York"
    elif "london" in p:
        slots["location"] = "London"
    elif "bangalore" in p:
        slots["location"] = "Bangalore"

    return slots

def select_variant(domain: str, prompt: str, slots: dict) -> list[dict]:
    """Select variant hint pack(s) based on user prompt and slots. Returns list (can be empty, single, or multiple)."""
    p = prompt.lower()

    if domain not in TEMPLATES or not TEMPLATES[domain].get("variants"):
        return []

    variants = TEMPLATES[domain]["variants"]
    selected = []

    # Domain-specific triggers
    if domain == "resume_guidance":
        # senior_lead trigger
        if any(w in p for w in ["senior", "lead", "principal", "staff", "director", "vp", "cto", "head of"]):
            selected.append(variants[0])  # senior_lead
        # technical_depth trigger
        elif any(w in p for w in ["architect", "technical", "system design", "performance", "optimization"]):
            selected.append(variants[1])  # technical_depth

    elif domain == "job_description":
        # tech_deep_dive trigger (tighter compliance checks)
        has_compliance = any(w in p for w in ["compliance", "security", "pci", "soc 2", "soc2", "hipaa"]) or re.search(r'\biso\s*27001\b', p)
        if slots.get("stack") or has_compliance:
            selected.append(variants[0])  # tech_deep_dive
        # remote_distributed trigger (both can be selected for JD!)
        if slots.get("location") == "Remote" or any(w in p for w in ["remote", "distributed", "timezone"]):
            selected.append(variants[1])  # remote_distributed

    elif domain == "job_resume_match":
        # quick_screen trigger (word boundaries)
        if re.search(r'\b(quick|screen(ing)?|brief|short|initial)\b', p):
            selected.append(variants[0])  # quick_screen

    elif domain == "recruiting_strategy":
        # streaming_aware trigger (more specific streaming terms)
        has_streaming = any(w in p for w in ["kafka", "streaming", "event stream", "event-driven", "kinesis", "pulsar", "reactive"])
        if has_streaming:
            selected.append(variants[0])  # streaming_aware
        # senior_executive trigger
        elif any(w in p for w in ["senior", "executive", "director", "vp", "c-level", "cto", "ceo"]):
            selected.append(variants[1])  # senior_executive

    elif domain == "ats_keywords":
        # role_specific trigger
        if slots.get("role"):
            selected.append(variants[0])  # role_specific

    # Enforce max_variants cap if specified in template
    if selected and domain in TEMPLATES:
        max_variants = TEMPLATES[domain].get("max_variants")
        if max_variants and len(selected) > max_variants:
            selected = selected[:max_variants]

    return selected

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

# ChatML builder (variant-aware, supports multiple variants)
def build_chatml(domain: str, user_prompt: str, variants: list[dict] = None, slots: dict | None = None, system_extra: str = "") -> str:
    """Build ChatML prompt with base template + optional variant hints (multiple) + slots + system_extra."""

    # Handle non-HR domains (small_talk, general_qna)
    if domain == "small_talk":
        sys_txt = "You are a friendly assistant. Respond warmly in 1–3 sentences."
        if system_extra:
            sys_txt += system_extra
        return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif domain == "general_qna":
        sys_txt = "You are a helpful assistant. Provide concise, accurate answers."
        if system_extra:
            sys_txt += system_extra
        return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    # HR domains - use TEMPLATES registry
    if domain not in TEMPLATES:
        # Fallback for unknown domains
        sys_txt = "You are a helpful assistant."
        if system_extra:
            sys_txt += system_extra
        return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    template = TEMPLATES[domain]

    # Prepend global HR style rules
    sys_txt = HR_STYLE_RULES

    # Inject slots context if present
    if slots and any(slots.values()):
        slot_parts = []
        if slots.get("role"):
            slot_parts.append(f"Target role: {slots['role']}")
        if slots.get("stack"):
            slot_parts.append(f"stack: {slots['stack']}")
        if slots.get("location"):
            slot_parts.append(f"location: {slots['location']}")
        if slot_parts:
            sys_txt += f"\nContext: {'; '.join(slot_parts)}."

    sys_txt += f"\n\n{template['base']}"

    # Add variant hints if provided (multiple variants supported)
    if variants:
        for variant in variants:
            hint = variant["hint"]
            # Inject slots into hint if present
            if slots:
                hint = hint.format(role=slots.get("role", "the role"),
                                  stack=slots.get("stack", "relevant technologies"),
                                  location=slots.get("location", ""))
            sys_txt += f"\n\nAdditional guidance: {hint}"

    # Add example if present
    if template.get("example"):
        sys_txt += f"\n\nExample:\n{template['example']}"

    # Add system_extra (e.g., clarifier on retry)
    if system_extra:
        sys_txt += system_extra

    # Add seeding
    seed = "1. " if template.get("seed") else ""

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

def _join_tokens(tokens: list) -> str:
    """Safely join token array (handles string tokens and dict tokens with 'text' key)."""
    if not tokens:
        return ""
    return "".join(
        t if isinstance(t, str) else (t.get("text", "") if isinstance(t, dict) else str(t))
        for t in tokens
    )

def merge_sampling(base_sampling: dict, template_sampling: dict) -> dict:
    """Merge template-level sampling overrides into base sampling.

    Special handling:
    - min_tokens: ensures max_tokens is at least min_tokens
    - temperature/top_p: clamped to [0, 1]
    - stop: coerced to list if needed

    Returns a new dict with merged values (does not modify inputs).
    """
    result = base_sampling.copy()

    for key, value in template_sampling.items():
        if key == "min_tokens":
            # Ensure max_tokens is at least min_tokens
            result["max_tokens"] = max(result.get("max_tokens", 0), value)
        elif key in ("temperature", "top_p"):
            # Clamp to [0, 1] for safety
            result[key] = max(0.0, min(1.0, float(value)))
        elif key == "stop":
            # Ensure stop is a list for decoder compatibility
            result[key] = value if isinstance(value, (list, tuple)) else [value]
        else:
            # Direct override for other keys (repetition_penalty, max_tokens, etc.)
            result[key] = value

    return result

def extract_text(payload: dict) -> str:
    """Extract text from response across common shapes (vLLM/raw/handler/chat).

    Supports:
    - Handler/completions: output.choices[0].text or tokens
    - Chat-style: output.choices[0].message.content or delta.content
    - Raw list format: output[0].choices[0].text/tokens/message.content
    - Plain string output
    - Root-level fallbacks: output.text, output.output_text, payload.text, payload.output_text
    """
    out = payload.get("output", [])

    # Helper to extract from a single choice dict
    def choice_text(choice: dict) -> str:
        # Chat-style content
        msg = choice.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return content
        # Completions-style
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text
        tokens = choice.get("tokens")
        if isinstance(tokens, list) and tokens:
            return _join_tokens(tokens)
        return ""

    # Handler endpoint format: dict with choices OR text
    if isinstance(out, dict):
        choices = out.get("choices", [])
        if isinstance(choices, list) and choices:
            ctext = choice_text(choices[0] or {})
            if ctext:
                return ctext
        # Common fallbacks on output root
        for key in ("text", "output_text"):
            val = out.get(key)
            if isinstance(val, str) and val:
                return val

    # Raw vLLM format: list with choices
    if isinstance(out, list) and out:
        first = out[0] or {}
        choices = first.get("choices", [])
        if isinstance(choices, list) and choices:
            ctext = choice_text(choices[0] or {})
            if ctext:
                return ctext
        # Fallbacks on the first item
        for key in ("text", "output_text"):
            val = first.get(key)
            if isinstance(val, str) and val:
                return val

    # Edge: output is a plain string
    if isinstance(out, str):
        return out

    # Root-level fallbacks
    for key in ("text", "output_text"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val

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
    """Call RunPod endpoint with ChatML prompt (with timeout retries)."""
    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": sampling,
        }
    }

    attempts = max(1, int(CLIENT_RETRY_ATTEMPTS) + 1)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                r = await client.post(URL_RUNSYNC, headers=HEADERS, json=payload)
                if r.status_code != 200:
                    raise Exception(f"Error {r.status_code}: {r.text[:300]}")
                return r.json()
            except httpx.TimeoutException as te:
                last_exc = te
                if i < attempts - 1:
                    delay = CLIENT_RETRY_BACKOFF_S * (2 ** i)
                    logger.debug(f"Timeout on attempt {i+1}/{attempts}. Backing off {delay:.2f}s…")
                    await asyncio.sleep(delay)
                    continue
                break
            except Exception as e:
                # Non-timeout errors are not retried; propagate
                last_exc = e
                break

        # Exhausted attempts
        if last_exc:
            raise last_exc
        raise Exception("Unknown error: request failed without exception")

async def generate_with_validation(domain: str, user_prompt: str, sampling: dict,
                                   enable_validation: bool, micro_retry: bool,
                                   variants: list[dict] = None, slots: dict | None = None) -> dict:
    """Generate response with optional validation and micro-retry."""
    # First attempt
    chat = build_chatml(domain, user_prompt, variants=variants, slots=slots)
    response = await call_endpoint(chat, sampling)
    text = extract_text(response)
    usage = extract_usage(response)
    exec_time = response.get("executionTime", 0)

    # Validate if enabled
    if enable_validation and domain in VALIDATORS:
        ok, issues = VALIDATORS[domain](text)

        # Micro-retry if validation fails and micro_retry enabled
        if not ok and micro_retry:
            clarifier = get_clarifier(domain, variants=variants)
            chat_retry = build_chatml(domain, user_prompt, variants=variants, slots=slots, system_extra=clarifier)
            response2 = await call_endpoint(chat_retry, sampling)
            text2 = extract_text(response2)
            ok2, issues2 = VALIDATORS[domain](text2)

            return {
                "text": text2,
                "ok": ok2,
                "issues": issues2,
                "exec_time": response2.get("executionTime", 0),
                "retried": True,
                "usage": extract_usage(response2),
                "variants": variants
            }

        return {
            "text": text,
            "ok": ok,
            "issues": issues,
            "exec_time": exec_time,
            "retried": False,
            "usage": usage,
            "variants": variants
        }

    # No validation
    return {
        "text": text,
        "ok": True,
        "issues": [],
        "exec_time": exec_time,
        "retried": False,
        "usage": usage,
        "variants": variants
    }

def get_clarifier(domain: str, variants: list[dict] = None) -> str:
    """Get clarifier text for micro-retry (variant-aware, supports multiple variants)."""
    base_clarifiers = {
        "resume_guidance": " Ensure 6–8 bullets with action verbs and end with: ATS Keywords: ...",
        "job_description": " Ensure sections: Summary + Responsibilities (6–8) + Requirements (6–8).",
        "job_resume_match": " Ensure Score (0–100) with explanation of why, Matches (5+), Gaps (3+), Next steps (3+). Explain the reasoning behind the score. Keep 100–150 words.",
        "recruiting_strategy": " Mention LinkedIn, referral, meetup, university, conference, GitHub, Stack Overflow and cadence weekly/monthly/quarterly.",
        "ats_keywords": " Provide 20–40 comma-separated keywords (80–120 words)."
    }

    clarifier = base_clarifiers.get(domain, "")

    # Add variant-specific clarifications (multiple variants supported)
    if variants:
        variant_clarifiers = {
            "senior_lead": " Focus on leadership impact and business outcomes.",
            "technical_depth": " Include specific technical details and architecture decisions.",
            "quick_screen": " Keep concise (80-120 words) focusing on must-haves.",
            "streaming_aware": (
                " Include Boolean sourcing patterns (JMS/ActiveMQ/RabbitMQ/Kinesis/Pulsar/Reactor/Netty), 4–6 channels with cadence, a personalized outreach opener, two screening prompts, and 1–2 metrics."
            ),
            "senior_executive": " Focus on executive search and longer hiring cycles.",
            "tech_deep_dive": " Include tech stack and compliance details.",
            "remote_distributed": " Emphasize remote work and distributed collaboration.",
            "role_specific": " Tailor keywords to the specific role."
        }
        for variant in variants:
            variant_name = variant["name"]
            if variant_name in variant_clarifiers:
                clarifier += variant_clarifiers[variant_name]

    return clarifier

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
        # Select variants and extract slots for HR domains
        variants = []
        slots = None
        if is_hr_domain:
            slots = extract_slots(prompt)
            variants = select_variant(domain, prompt, slots)

            # Debug logging (controlled by logger level)
            variant_names = [v["name"] for v in variants] if variants else []
            logger.debug(f"Domain: {domain}")
            logger.debug(f"Selected variants: {variant_names}")
            logger.debug(f"Extracted slots: {slots}")

        # Merge template-level sampling overrides if present
        if domain in TEMPLATES and TEMPLATES[domain].get("sampling"):
            sampling = merge_sampling(sampling, TEMPLATES[domain]["sampling"])

        result = await generate_with_validation(
            domain,
            prompt,
            sampling,
            actual_validation,
            actual_retry,
            variants=variants,
            slots=slots,
        )
    except Exception as e:
        # Chainlit Message.update() does not accept keyword args; set content then update
        is_timeout = isinstance(e, (httpx.TimeoutException, httpx.ReadTimeout))
        err_msg = (
            "❌ Request timed out. Try again shortly or increase CLIENT_TIMEOUT_S / CLIENT_RETRY_ATTEMPTS."
            if is_timeout
            else f"❌ Request error: {e}"
        )
        thinking.content = err_msg
        await thinking.update()
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

    # Add variant info if present (support multiple variants)
    variants_info = result.get("variants")
    if variants_info:
        variant_names = ", ".join([v["name"] for v in variants_info])
        status_parts.append(f"🧩 Variant: {variant_names}")

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

    # Stream by chunks, preserving whitespace and formatting
    chunks = re.split(r'(\s+)', text)
    for chunk in chunks:
        if chunk:  # Skip empty strings
            await asyncio.sleep(0)  # yield
            await out.stream_token(chunk)

    # Add status footer
    await out.stream_token(status)
    await out.update()

def detect_domain(prompt: str) -> str | None:
    """Detect HR domain from user prompt (heuristic fallback)."""
    p = prompt.lower().strip()

    # Small talk (greetings) - check for exact "hi" or "hi " at word boundaries
    if any(w in p for w in ["hello", "hey", "how are you", "good morning", "good evening"]):
        return "small_talk"
    # Exact match for "hi" at start or with word boundary
    if p == "hi" or p.startswith("hi ") or " hi " in p or p.endswith(" hi"):
        return "small_talk"

    # ATS keywords (use word boundaries to avoid "stats" false positives)
    if re.search(r'\bats\b', p) or any(w in p for w in ["applicant tracking", "ats keyword", "resume keyword", "applicant tracking system"]):
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
