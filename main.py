from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI()

# Request contract
class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)

# Client-facing response contract
class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = Field(min_length=1)


class CompletionChoice(BaseModel):
    index: int = Field(ge=0)
    message: AssistantMessage
    finish_reason: Literal["stop"]


class ChatResponse(BaseModel):
    id: str = Field(min_length=1)
    object: Literal["chat.completion"]
    created: int = Field(ge=0)
    model: str = Field(min_length=1)
    choices: list[CompletionChoice] = Field(min_length=1)


# simulated provider implementations
def mock_provider_a(messages: list[Message]) -> str:
    return f"Mock A received: {messages[-1].content}"

def mock_provider_b(messages: list[Message]) -> str:
    return f"Mock B received: {messages[-1].content}"


# provider registry
ProviderFunction = Callable[[list[Message]], str]

PROVIDERS: dict[str, ProviderFunction] = {
    "provider-a": mock_provider_a,
    "provider-b": mock_provider_b,
}

# Model routing config
MODEL_ROUTES = {
    "mock-model": "provider-a",
    "mock-model-b": "provider-b",
}

# Provider output boundary
def validate_provider_output(raw_output: object) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise HTTPException(
            status_code=502,
            detail="Invalid provider response",
        )

    return raw_output


# Gateway endpoint
@app.post("/v1/chat/completions", response_model=ChatResponse,)
def chat_completions(request: ChatRequest) -> ChatResponse:
    provider_name = MODEL_ROUTES.get(request.model)

    if provider_name is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )
    provider = PROVIDERS.get(provider_name)

    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Configured provider unavailable",
        )

    raw_output = provider(request.messages)
    completion_text = validate_provider_output(raw_output)

    return ChatResponse(
        id="chatcmpl-mock",
        object="chat.completion",
        created=0,
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                message=AssistantMessage(
                    content=completion_text,
                ),
                finish_reason="stop",
            )
        ],
    )