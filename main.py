from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI()


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["mock-model"]
    messages: list[Message] = Field(min_length=1)


def mock_provider() -> str:
    return "Hello from the mock provider."


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": mock_provider(),
                },
                "finish_reason": "stop",
            }
        ],
    }