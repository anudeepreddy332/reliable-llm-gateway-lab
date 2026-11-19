
from dataclasses import dataclass
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from reliability import (
    DeadlineExceeded,
    ProviderAttempt,
    execute_plan,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(main.app) as test_client:
        yield test_client


def payload(model: str = "mock-retry") -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }


@dataclass
class FakeTime:
    now: float = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize(
    "kind",
    [
        main.ProviderFailureKind.RATE_LIMIT,
        main.ProviderFailureKind.UNAVAILABLE,
    ],
)
def test_transient_failure_retries_once(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: main.ProviderFailureKind,
) -> None:
    calls: list[int] = []

    def provider(messages: list[main.Message]) -> str:
        calls.append(1)

        if len(calls) == 1:
            raise main.ProviderExecutionError(
                kind, "Transient failure"
            )

        return "Recovered"

    monkeypatch.setitem(main.PROVIDERS, "provider-a", provider)

    response = client.post(
        "/v1/chat/completions", json=payload()
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Recovered"
    )
    assert len(calls) == 2


def test_retry_limit_enforced(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def provider(messages: list[main.Message]) -> str:
        calls.append(1)
        raise main.ProviderExecutionError(
            main.ProviderFailureKind.UNAVAILABLE,
            "Unavailable",
        )

    monkeypatch.setitem(main.PROVIDERS, "provider-a", provider)

    response = client.post(
        "/v1/chat/completions", json=payload()
    )

    assert response.status_code == 503
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        (main.ProviderFailureKind.TIMEOUT, 504),
        (main.ProviderFailureKind.CONNECTION, 502),
    ],
)
def test_ambiguous_failures_are_not_retried(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    kind: main.ProviderFailureKind,
    status: int,
) -> None:
    calls: list[int] = []

    def provider(messages: list[main.Message]) -> str:
        calls.append(1)
        raise main.ProviderExecutionError(
            kind, "Ambiguous outcome"
        )

    monkeypatch.setitem(main.PROVIDERS, "provider-a", provider)

    response = client.post(
        "/v1/chat/completions", json=payload()
    )

    assert response.status_code == status
    assert len(calls) == 1


def test_remaining_budget_shrinks_after_backoff() -> None:
    fake = FakeTime()
    budgets: list[float] = []

    def first(remaining: float) -> str:
        budgets.append(remaining)
        raise RuntimeError("Eligible simulated failure")

    def second(remaining: float) -> str:
        budgets.append(remaining)
        return "Recovered"

    result = execute_plan(
        (
            ProviderAttempt("a", first),
            ProviderAttempt("b", second),
        ),
        can_continue=lambda exc: isinstance(exc, RuntimeError),
        total_timeout_seconds=1.0,
        backoff_seconds=0.2,
        clock=fake.clock,
        sleep=fake.sleep,
    )

    assert result.content == "Recovered"
    assert result.attempts == 2
    assert budgets == pytest.approx([1.0, 0.8])


def test_deadline_prevents_second_attempt() -> None:
    fake = FakeTime()
    calls: list[str] = []

    def slow_failure(remaining: float) -> str:
        calls.append("first")
        fake.now += 0.96
        raise RuntimeError("Simulated failure")

    def second(remaining: float) -> str:
        calls.append("second")
        return "Should not execute"

    with pytest.raises(DeadlineExceeded):
        execute_plan(
            (
                ProviderAttempt("a", slow_failure),
                ProviderAttempt("b", second),
            ),
            can_continue=lambda exc: True,
            total_timeout_seconds=1.0,
            backoff_seconds=0.05,
            clock=fake.clock,
            sleep=fake.sleep,
        )

    assert calls == ["first"]


def test_late_completion_is_rejected() -> None:
    fake = FakeTime()

    def late_provider(remaining: float) -> str:
        fake.now += 1.1
        return "Late completion"

    with pytest.raises(DeadlineExceeded):
        execute_plan(
            (ProviderAttempt("a", late_provider),),
            can_continue=lambda exc: False,
            total_timeout_seconds=1.0,
            clock=fake.clock,
            sleep=fake.sleep,
        )


@pytest.mark.parametrize(
    ("max_attempts", "timeout", "backoff"),
    [
        (0, 1.0, 0.0),
        (2, 0.0, 0.0),
        (2, 1.0, -1.0),
    ],
)
def test_invalid_execution_policy_rejected(
    max_attempts: int,
    timeout: float,
    backoff: float,
) -> None:
    with pytest.raises(ValueError):
        execute_plan(
            (ProviderAttempt("a", lambda remaining: "OK"),),
            can_continue=lambda exc: False,
            max_attempts=max_attempts,
            total_timeout_seconds=timeout,
            backoff_seconds=backoff,
        )


@pytest.mark.parametrize(
    "model",
    ["deepseek-flash", "cohere-command-r7b"],
)
def test_real_adapter_receives_timeout_budget(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    observed: list[dict[str, float]] = []

    if model == "deepseek-flash":
        provider_module = main.deepseek
        key_name = "DEEPSEEK_API_KEY"
        upstream = {
            "id": "test-123",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "READY",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    else:
        provider_module = main.cohere
        key_name = "COHERE_API_KEY"
        upstream = {
            "id": "test-123",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "READY"}
                ],
            },
        }

    monkeypatch.setenv("GATEWAY_ENABLE_LIVE_PROVIDER", "1")
    monkeypatch.setenv(key_name, "test-only-key")

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.extensions["timeout"])
        return httpx.Response(200, json=upstream)

    def mock_client() -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=30.0,
            trust_env=False,
        )

    monkeypatch.setattr(
        provider_module, "create_client", mock_client
    )

    response = client.post(
        "/v1/chat/completions",
        json=payload(model),
    )

    assert response.status_code == 200
    assert len(observed) == 1

    timeout_values = observed[0]

    assert 0 < timeout_values["read"] <= 20.0
    assert 0 < timeout_values["connect"] <= 5.0