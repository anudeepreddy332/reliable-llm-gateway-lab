import httpx
from pydantic import BaseModel, Field
from typing import Literal

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-flash"

class DeepSeekResponseError(Exception):
    """Provider returned an unsupported or invalid completion."""


class UpstreamMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class UpstreamChoice(BaseModel):
    index: Literal[0]
    message: UpstreamMessage
    finish_reason: Literal["stop"]


class UpstreamCompletion(BaseModel):
    id: str = Field(min_length=1)
    object: Literal["chat.completion"]
    created: int = Field(ge=0)
    model: str = Field(min_length=1)
    choices: list[UpstreamChoice] = Field(
        min_length=1,
        max_length=1,
    )

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
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "stream": False,
            "thinking": {"type": "disabled"},
            "max_tokens": 96,
        },
    )
    response.raise_for_status()

    try:
        completion = UpstreamCompletion.model_validate(response.json())

    except ValueError as exc:
        raise DeepSeekResponseError(
            "Invalid upstream completion"
        ) from exc

    if completion.model != DEEPSEEK_MODEL:
        raise DeepSeekResponseError(
            "Unexpected upstream model"
        )

    return completion.choices[0].message.content