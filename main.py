from typing import Callable, Literal
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import os
import httpx
from providers import deepseek, cohere
from functools import partial
from reliability import ProviderAttempt, execute_plan, DeadlineExceeded
from dataclasses import dataclass


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

# Provider failure classification
class ProviderFailureKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CONNECTION = "connection"

class ProviderExecutionError(Exception):
    def __init__(
        self,
        kind: ProviderFailureKind,
        message: str,
    ) -> None:
        self.kind = kind
        super().__init__(message)


PROVIDER_FAILURE_HTTP: dict[
    ProviderFailureKind, tuple[int, str]
] = {
    ProviderFailureKind.RATE_LIMIT: (
        429, "Upstream provider rate limited"
    ),
    ProviderFailureKind.UNAVAILABLE: (
        503, "Upstream provider unavailable"
    ),
    ProviderFailureKind.TIMEOUT: (
        504, "Upstream provider timed out"
    ),
    ProviderFailureKind.CONNECTION: (
        502, "Upstream connection failed"
    ),
}


# simulated provider implementations
def mock_provider_a(messages: list[Message]) -> str:
    return f"Mock A received: {messages[-1].content}"

def mock_provider_b(messages: list[Message]) -> str:
    return f"Mock B received: {messages[-1].content}"


def deepseek_provider(messages: list[Message],
                      *,
                      timeout_seconds: float | None = None,
                      ) -> str:
    if os.getenv("GATEWAY_ENABLE_LIVE_PROVIDER") != "1":
        raise HTTPException(
            status_code=503,
            detail="Live provider access disabled",
        )

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="DeepSeek provider not configured",
        )

    provider_messages = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]

    try:
        with deepseek.create_client() as client:
            return deepseek.complete(
                provider_messages,
                api_key,
                client=client,
                timeout_seconds=timeout_seconds,
            )

    except httpx.TimeoutException as exc:
        raise ProviderExecutionError(
            ProviderFailureKind.TIMEOUT,
            "DeepSeek request timed out",
        ) from exc

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status == 429:
            kind = ProviderFailureKind.RATE_LIMIT
        elif status in (408, 504):
            kind = ProviderFailureKind.TIMEOUT
        elif status in (500, 502, 503):
            kind = ProviderFailureKind.UNAVAILABLE
        else:
            raise HTTPException(
                status_code=502,
                detail="Upstream provider request rejected",
            ) from exc

        raise ProviderExecutionError(
            kind,
            "DeepSeek upstream HTTP failure",
        ) from exc

    except httpx.RequestError as exc:
        raise ProviderExecutionError(
            ProviderFailureKind.CONNECTION,
            "DeepSeek transport failure",
        ) from exc

    except deepseek.DeepSeekResponseError as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid provider response",
        ) from exc


# Cohere provider
def cohere_provider(messages: list[Message],
                    *,
                    timeout_seconds: float | None = None,) -> str:
    if os.getenv("GATEWAY_ENABLE_LIVE_PROVIDER") != "1":
        raise HTTPException(
            status_code=503,
            detail="Live provider access disabled",
        )

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Cohere provider not configured",
        )

    provider_messages = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]

    try:
        with cohere.create_client() as client:
            return cohere.complete(
                provider_messages,
                api_key,
                client=client,
                timeout_seconds=timeout_seconds,
            )

    except httpx.TimeoutException as exc:
        raise ProviderExecutionError(
            ProviderFailureKind.TIMEOUT,
            "Cohere request timed out",
        ) from exc

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status == 429:
            kind = ProviderFailureKind.RATE_LIMIT
        elif status in (408, 504):
            kind = ProviderFailureKind.TIMEOUT
        elif status in (500, 502, 503):
            kind = ProviderFailureKind.UNAVAILABLE
        else:
            raise HTTPException(
                status_code=502,
                detail="Upstream provider request rejected",
            ) from exc

        raise ProviderExecutionError(
            kind,
            "Cohere upstream HTTP failure",
        ) from exc

    except httpx.RequestError as exc:
        raise ProviderExecutionError(
            ProviderFailureKind.CONNECTION,
            "Cohere transport failure",
        ) from exc

    except cohere.CohereResponseError as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid provider response",
        ) from exc


# provider registry
ProviderFunction = Callable[[list[Message]], str]


@dataclass(frozen=True)
class BudgetedProvider:
    invoke: Callable[[list[Message], float | None], str]

    def __call__(self, messages: list[Message]) -> str:
        return self.invoke(messages, None)

    def with_budget(
        self,
        messages: list[Message],
        remaining_seconds: float,
    ) -> str:
        return self.invoke(messages, remaining_seconds)


PROVIDERS: dict[str, ProviderFunction] = {
    "provider-a": mock_provider_a,
    "provider-b": mock_provider_b,
    "deepseek": BudgetedProvider(
        lambda messages, seconds: deepseek_provider(
            messages,
            timeout_seconds=seconds,
        )
    ),
    "cohere": BudgetedProvider(
        lambda messages, seconds: cohere_provider(
            messages,
            timeout_seconds=seconds,
        )
    ),
}

# Model routing config
MODEL_ROUTES: dict[str, str] = {
    "mock-model": "provider-a",
    "mock-model-b": "provider-b",
    "deepseek-flash": "deepseek",
    "cohere-command-r7b": "cohere",
}

# Explicit fallback routes.
# No automatic real-provider fallback is enabled.
FALLBACK_ROUTES: dict[str, tuple[str, ...]] = {
    "mock-fallback": ("provider-a", "provider-b"),
    "mock-retry": ("provider-a", "provider-a"),
}

# Provider output boundary
def validate_provider_output(raw_output: object) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise HTTPException(
            status_code=502,
            detail="Invalid provider response",
        )

    return raw_output


def invoke_validated(
    provider: ProviderFunction,
    messages: list[Message],
    remaining_seconds: float,
) -> str:

    if isinstance(provider, BudgetedProvider):
        raw_output = provider.with_budget(
            messages,
            remaining_seconds,
        )

    else:
        raw_output = provider(messages)

    return validate_provider_output(raw_output)

def can_failover(exc: Exception) -> bool:
    return (
        isinstance(exc, ProviderExecutionError)
        and exc.kind in {
            ProviderFailureKind.RATE_LIMIT,
            ProviderFailureKind.UNAVAILABLE,
        }
    )

# Gateway endpoint
@app.post("/v1/chat/completions", response_model=ChatResponse,)
def chat_completions(request: ChatRequest) -> ChatResponse:
    provider_names = FALLBACK_ROUTES.get(request.model)

    if provider_names is None:
        primary_name = MODEL_ROUTES.get(request.model)

        if primary_name is None:
            raise HTTPException(
                status_code=404,
                detail="Model not found"
            )

        provider_names = (primary_name,)

    attempts: list[ProviderAttempt] = []

    for provider_name in provider_names:
        provider = PROVIDERS.get(provider_name)

        if provider is None:
            raise HTTPException(
                status_code=503,
                detail="Configured provider unavailable",
            )

        attempts.append(
            ProviderAttempt(
                name=provider_name,
                execute=partial(
                    invoke_validated,
                    provider,
                    request.messages,
                ),
            )
        )

    try:
        result = execute_plan(
            tuple(attempts),
            can_continue=can_failover,
            max_attempts=2,
            total_timeout_seconds=20.0,
            backoff_seconds=0.05,
        )

    except DeadlineExceeded as exc:
        raise HTTPException(
            status_code=504,
            detail="Gateway execution deadline exceeded",
        ) from exc

    except ProviderExecutionError as exc:
        status_code, detail = PROVIDER_FAILURE_HTTP[exc.kind]

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    return ChatResponse(
        id="chatcmpl-mock",
        object="chat.completion",
        created=0,
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                message=AssistantMessage(
                    content=result.content,
                ),
                finish_reason="stop",
            )
        ],
    )