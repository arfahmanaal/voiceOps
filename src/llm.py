import os
import re
import time
from collections import deque

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")
MODELS = [LLM_MODEL] + [name.strip() for name in os.getenv("LLM_FALLBACK_MODEL", "").split(",") if name.strip()]
TPM = int(os.getenv("LLM_TPM", "7000"))
MAX_WAIT = 120
UNITS = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}

usage = {name: deque() for name in MODELS}
blocked = {}


def retry_after(error):
    match = re.search(r"try again in ([0-9hms.]+)", str(error))
    if not match:
        return None
    total = sum(float(value) * UNITS[unit] for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", match.group(1)))
    return total or None


def pace(model, estimate):
    window = usage[model]
    while True:
        now = time.time()
        while window and now - window[0][0] > 60:
            window.popleft()
        used = sum(tokens for _, tokens in window)
        if not window or used + estimate <= TPM:
            return
        wait = 60 - (now - window[0][0]) + 0.5
        print(f"[pacing] {model}: waiting {wait:.0f}s for the per-minute token limit", flush=True)
        time.sleep(wait)


def chat(system, messages, temperature=0.2, max_tokens=1500):
    payload = [{"role": "system", "content": system}] + messages
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    estimate = sum(len(item["content"]) for item in payload) // 2 + 700
    for attempt in range(10):
        errors = []
        for model in MODELS:
            if blocked.get(model, 0) > time.time():
                continue
            pace(model, estimate)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra,
                )
                tokens = getattr(getattr(response, "usage", None), "total_tokens", None) or estimate
                usage[model].append((time.time(), tokens))
                return (response.choices[0].message.content or "").strip()
            except RateLimitError as error:
                wait = retry_after(error)
                if wait is not None and wait > MAX_WAIT:
                    blocked[model] = time.time() + wait
                    print(f"[llm limit] {model} unavailable for about {wait / 60:.0f} min: {str(error)[:220]}", flush=True)
                errors.append((wait, error))
            except (APIConnectionError, InternalServerError) as error:
                errors.append((5 * (attempt + 1), error))
        live = [name for name in MODELS if blocked.get(name, 0) <= time.time()]
        if not live:
            earliest = min(blocked.values()) - time.time()
            raise RuntimeError(
                f"Every configured model is limited. The earliest one frees up in about {earliest / 60:.0f} minutes. "
                "Rerun the same command then; finished work is saved."
            )
        waits = [wait for wait, _ in errors if wait is not None]
        wait = min(min(waits) if waits else 10 * (attempt + 1), MAX_WAIT)
        print(f"[llm limit] waiting {wait + 1:.0f}s", flush=True)
        time.sleep(wait + 1)
    raise RuntimeError(f"LLM retries exhausted. Last error: {errors[-1][1] if errors else 'unknown'}")