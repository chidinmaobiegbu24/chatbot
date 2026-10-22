import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly chatbot. Keep answers clear and concise.",
)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
VALID_THINKING_LEVELS = {"none", "minimal", "low", "medium", "high"}


def get_client() -> OpenAI:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        print(f"Add it to {ENV_FILE} like:  GEMINI_API_KEY=AIza...")
        sys.exit(1)
    if THINKING_LEVEL not in VALID_THINKING_LEVELS:
        print(
            f"Error: invalid GEMINI_THINKING_LEVEL '{THINKING_LEVEL}'. "
            f"Valid values are: {', '.join(sorted(VALID_THINKING_LEVELS))}."
        )
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def describe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 400:
        return (
            "Bad request (400) - the model or parameters were rejected. "
            "Check GEMINI_THINKING_LEVEL is one of: "
            f"{', '.join(sorted(VALID_THINKING_LEVELS))}."
        )
    if status == 401:
        return "Invalid API key (401). Check GEMINI_API_KEY in .env."
    if status == 403:
        return "Access denied (403). The Gemini API may not be enabled for this key."
    if status == 404:
        return f"Model not found (404): {MODEL}. Set GEMINI_MODEL to a valid id."
    if status == 429:
        return "Rate limited (429). Wait a moment and try again."
    if status in RETRYABLE_STATUS:
        return f"Service unavailable ({status}). Try again in a moment."
    if isinstance(exc, APIConnectionError):
        return "Could not reach the Gemini API (network or timeout error)."
    return str(exc)


def create_completion(client: OpenAI, messages: list[dict]) -> object:
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                reasoning_effort=THINKING_LEVEL,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in RETRYABLE_STATUS or isinstance(
                exc, APIConnectionError
            )
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            delay = 2**attempt
            label = status if status is not None else "network error"
            print(f"  {label} from Gemini, retrying in {delay}s...")
            time.sleep(delay)


def chat() -> None:
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatbot ready (model: {MODEL}). Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})
        try:
            response = create_completion(client, messages)
        except Exception as exc:
            messages.pop()
            print(f"API error: {describe_error(exc)}\n")
            continue

        reply = response.choices[0].message.content or ""
        if not reply.strip():
            messages.pop()
            print("Bot: (empty reply from the model, try rephrasing.)\n")
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    chat()
