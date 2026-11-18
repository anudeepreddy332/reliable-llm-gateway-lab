from typing import Literal

import httpx
from pydantic import BaseModel, Field


COHERE_URL = "https://api.cohere.com/v2/chat"
COHERE_MODEL = "command-r7b-12-2024"

class CohereResponseError(Exception):
    """Cohere returned an unsupported or invalid completion."""


class ContentBlock(BaseModel):
    type: Literal["text"]
    text: str

class CohereMessage(BaseModel):
    role: Literal["assistant"]
    content: list[ContentBlock] = Field(min_length=1)

class CohereCompletion(BaseModel):
    id: str = Field(min_length=1)
    finish_reason: Literal["COMPLETE"]
    message: CohereMessage

def create_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=5.0),
        follow_redirects=False,
        trust_env=False,
    )

def complete(
    messages: list[dict[str, str]],
    api_key: str,
    *,
    client: httpx.Client,
) -> str:
    response = client.post(
        COHERE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": COHERE_MODEL,
            "messages": messages,
            "stream": False,
            "max_tokens": 128,
        },
    )

    response.raise_for_status()

    try:
        completion = CohereCompletion.model_validate(
            response.json()
        )
    except ValueError as exc:
        raise CohereResponseError(
            "Invalid Cohere completion"
        ) from exc

    content = "".join(
        block.text for block in completion.message.content
    )

    if not content.strip():
        raise CohereResponseError(
            "Empty Cohere completion"
        )

    return content