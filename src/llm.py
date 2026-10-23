import os
import time

from openai import OpenAI, RateLimitError

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")


def chat(system, messages, temperature=0.2, max_tokens=1500):
    payload = [{"role": "system", "content": system}] + messages
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("LLM rate limit: retries exhausted")