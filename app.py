import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI, APIError

app = FastAPI(title="Gemini Chatbot API")

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Chatbot API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured",
        )

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    client = OpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            messages=[
                {
                    "role": "system",
                    "content": os.getenv(
                        "SYSTEM_PROMPT",
                        "You are a helpful, friendly chatbot. "
                        "Keep answers clear and concise.",
                    ),
                },
                {"role": "user", "content": request.message},
            ],
        )

        return {
            "reply": response.choices[0].message.content or ""
        }

    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail="The Gemini service could not complete your request.",
        ) from exc
