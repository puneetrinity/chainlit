import os, json, asyncio
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
    "max_tokens": 180,            # domain default; override per request if needed
    "temperature": 0.3,
    "top_p": 0.9
}

DEFAULT_FLAGS = {
    "enable_validation": True,
    "block_low_trust_intents": True   # keep salary/market blocked by default
}

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# HR Task Templates
HR_TASKS = {
    "resume_guidance": {
        "label": "📄 Resume Guidance",
        "prompt": "I need help improving my resume. Please provide specific, actionable guidance on formatting, content, and how to highlight my achievements effectively."
    },
    "job_description": {
        "label": "📝 Job Description Help",
        "prompt": "Help me write or improve a job description. Include key responsibilities, required qualifications, and how to make it attractive to candidates."
    },
    "ats_keywords": {
        "label": "🔍 ATS Keywords",
        "prompt": "What keywords should I include in my resume/job posting to pass ATS (Applicant Tracking Systems)? Focus on relevant skills and qualifications."
    },
    "job_resume_match": {
        "label": "🎯 Job-Resume Match",
        "prompt": "Analyze how well a resume matches a job description. Identify gaps, strengths, and provide recommendations for improvement."
    },
    "recruiting_strategy": {
        "label": "📊 Recruiting Strategy",
        "prompt": "Provide a recruiting strategy for attracting top talent. Include sourcing channels, outreach techniques, and how to evaluate candidates effectively."
    }
}

def extract_text(payload: dict) -> str:
    # Supports both text and tokens shape
    out = payload.get("output", {})
    if isinstance(out, dict):
        choices = (out.get("choices") or [{}])
        ch = choices[0] if choices else {}
        return ch.get("text") or (ch.get("tokens") or [""])[0] or ""
    if isinstance(out, list) and out:
        ch = (out[0].get("choices") or [{}])[0]
        return ch.get("text") or (ch.get("tokens") or [""])[0] or ""
    return ""

def extract_validation(payload: dict) -> dict | None:
    out = payload.get("output", {})
    return out.get("validation") if isinstance(out, dict) else None

def extract_usage(payload: dict) -> dict | None:
    out = payload.get("output", {})
    return out.get("usage") if isinstance(out, dict) else None

def extract_block(payload: dict) -> tuple[bool, str | None]:
    out = payload.get("output", {})
    return bool(out.get("blocked", False)), out.get("message")

@cl.on_chat_start
async def start():
    # Initialize settings in user session
    cl.user_session.set("settings", {
        "temperature": DEFAULT_SAMPLING["temperature"],
        "top_p": DEFAULT_SAMPLING["top_p"],
        "max_tokens": DEFAULT_SAMPLING["max_tokens"],
        "enable_validation": DEFAULT_FLAGS["enable_validation"],
        "block_low_trust_intents": DEFAULT_FLAGS["block_low_trust_intents"]
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
            max=500,
            step=10,
            description="Maximum length of response"
        ),
        Switch(
            id="enable_validation",
            label="Enable Validation",
            initial=DEFAULT_FLAGS["enable_validation"],
            description="Validate prompts against HR domain policies"
        ),
        Switch(
            id="block_low_trust_intents",
            label="Block Low-Trust Intents",
            initial=DEFAULT_FLAGS["block_low_trust_intents"],
            description="Block salary/market research queries (untrusted in this domain)"
        )
    ]).send()

    # Create HR task action buttons
    actions = [
        cl.Action(name=task_id, payload=task_data["prompt"], label=task_data["label"])
        for task_id, task_data in HR_TASKS.items()
    ]

    # Send welcome message with actions
    await cl.Message(
        content="👋 **Welcome to HR Career Assistant!**\n\n"
                "Ask career or recruiting questions, or use quick actions below:\n\n"
                "⚙️ Adjust settings in the sidebar to customize responses.\n"
                "🔧 Use `temp=0.2` inline to override temperature for a single query.",
        actions=actions
    ).send()

@cl.on_settings_update
async def settings_update(settings):
    """Handle settings changes."""
    cl.user_session.set("settings", settings)
    await cl.Message(content=f"✅ Settings updated: temp={settings['temperature']}, top_p={settings['top_p']}, "
                              f"validation={settings['enable_validation']}, block_low_trust={settings['block_low_trust_intents']}").send()

@cl.action_callback("resume_guidance")
async def on_resume_guidance(action: cl.Action):
    """Handle Resume Guidance action."""
    await process_hr_task(action.payload)

@cl.action_callback("job_description")
async def on_job_description(action: cl.Action):
    """Handle Job Description action."""
    await process_hr_task(action.payload)

@cl.action_callback("ats_keywords")
async def on_ats_keywords(action: cl.Action):
    """Handle ATS Keywords action."""
    await process_hr_task(action.payload)

@cl.action_callback("job_resume_match")
async def on_job_resume_match(action: cl.Action):
    """Handle Job-Resume Match action."""
    await process_hr_task(action.payload)

@cl.action_callback("recruiting_strategy")
async def on_recruiting_strategy(action: cl.Action):
    """Handle Recruiting Strategy action."""
    await process_hr_task(action.payload)

async def process_hr_task(prompt: str):
    """Process HR task with current settings."""
    settings = cl.user_session.get("settings")

    sampling = {
        "max_tokens": settings.get("max_tokens", DEFAULT_SAMPLING["max_tokens"]),
        "temperature": settings.get("temperature", DEFAULT_SAMPLING["temperature"]),
        "top_p": settings.get("top_p", DEFAULT_SAMPLING["top_p"])
    }

    flags = {
        "enable_validation": settings.get("enable_validation", DEFAULT_FLAGS["enable_validation"]),
        "block_low_trust_intents": settings.get("block_low_trust_intents", DEFAULT_FLAGS["block_low_trust_intents"])
    }

    await send_query(prompt, sampling, flags)

async def send_query(prompt: str, sampling: dict, flags: dict):
    """Core function to send query to RunPod and stream response."""
    if not ENDPOINT_ID or not API_KEY:
        await cl.Message(content="Server missing RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY.").send()
        return

    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": sampling,
            **flags
        }
    }

    thinking = cl.Message(content="Thinking…")
    await thinking.send()

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(URL_RUNSYNC, headers=HEADERS, json=payload)
            if r.status_code != 200:
                await thinking.update(content=f"Error {r.status_code}: {r.text[:300]}")
                return
            body = r.json()
    except Exception as e:
        await thinking.update(content=f"Request error: {e}")
        return

    blocked, block_msg = extract_block(body)
    if blocked:
        await thinking.update(content=block_msg or "Blocked by policy.")
        return

    text = extract_text(body)
    validation = extract_validation(body)
    usage = extract_usage(body)

    # Remove thinking message before streaming
    await thinking.remove()

    # Simulate token streaming from final text
    out = cl.Message(content="")
    await out.send()
    # Stream by words; adjust chunking per preference
    for token in (text or "").split():
        await asyncio.sleep(0)  # yield
        await out.stream_token(token + " ")
    await out.update()

    # Append validation/usage info as a side message
    meta_lines = []
    if validation:
        intent = validation.get("intent", "-")
        valid = validation.get("valid", None)
        issues = validation.get("issues", [])
        meta_lines.append(f"**Validation:** intent={intent}, valid={valid}")
        if issues:
            meta_lines.append(f"**Issues:** {'; '.join(issues[:4])}")
    if usage:
        meta_lines.append(f"**Usage:** input={usage.get('input')} tokens, output={usage.get('output')} tokens")
    if meta_lines:
        await cl.Message(content="\n".join(meta_lines)).send()

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
        "block_low_trust_intents": DEFAULT_FLAGS["block_low_trust_intents"]
    }

    sampling = {
        "max_tokens": settings.get("max_tokens", DEFAULT_SAMPLING["max_tokens"]),
        "temperature": settings.get("temperature", DEFAULT_SAMPLING["temperature"]),
        "top_p": settings.get("top_p", DEFAULT_SAMPLING["top_p"])
    }

    # Inline override: temp=0.2 in message
    if "temp=" in prompt:
        try:
            val = float(prompt.split("temp=")[1].split()[0])
            sampling["temperature"] = max(0.0, min(1.0, val))
        except Exception:
            pass

    flags = {
        "enable_validation": settings.get("enable_validation", DEFAULT_FLAGS["enable_validation"]),
        "block_low_trust_intents": settings.get("block_low_trust_intents", DEFAULT_FLAGS["block_low_trust_intents"])
    }

    await send_query(prompt, sampling, flags)
