from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import main

@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(main.app) as test_client:
        yield test_client

def request_payload(
    model: str = "mock-fallback",
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
    }

def test_primary_success_skips_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def primary(messages: list[main.Message]) -> str:
        calls.append("primary")
        return "Primary succeeded"

    def secondary(messages: list[main.Message]) -> str:
        calls.append("secondary")
        return "Secondary succeeded"

    monkeypatch.setitem(main.PROVIDERS, "provider-a", primary)
    monkeypatch.setitem(main.PROVIDERS, "provider-b", secondary)

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(),
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Primary succeeded"
    )
    assert calls == ["primary"]


@pytest.mark.parametrize(
    "failure_kind",
    [
        main.ProviderFailureKind.RATE_LIMIT,
        main.ProviderFailureKind.UNAVAILABLE,
    ],
)
def test_eligible_failure_uses_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: main.ProviderFailureKind,
) -> None:
    calls: list[str] = []

    def primary(messages: list[main.Message]) -> str:
        calls.append("primary")
        raise main.ProviderExecutionError(
            failure_kind,
            "Private upstream diagnostic",
        )

    def secondary(messages: list[main.Message]) -> str:
        calls.append("secondary")
        return "Fallback succeeded"

    monkeypatch.setitem(main.PROVIDERS, "provider-a", primary)
    monkeypatch.setitem(main.PROVIDERS, "provider-b", secondary)

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(),
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Fallback succeeded"
    )
    assert calls == ["primary", "secondary"]

@pytest.mark.parametrize(
    ("failure_kind", "expected_status"),
    [
        (main.ProviderFailureKind.TIMEOUT, 504),
        (main.ProviderFailureKind.CONNECTION, 502),
    ],
)
def test_uncertain_failure_does_not_replay(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: main.ProviderFailureKind,
    expected_status: int,
) -> None:
    calls: list[str] = []

    def primary(messages: list[main.Message]) -> str:
        calls.append("primary")
        raise main.ProviderExecutionError(
            failure_kind,
            "Uncertain upstream outcome",
        )

    def secondary(messages: list[main.Message]) -> str:
        calls.append("secondary")
        return "Should not execute"

    monkeypatch.setitem(main.PROVIDERS, "provider-a", primary)
    monkeypatch.setitem(main.PROVIDERS, "provider-b", secondary)

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(),
    )

    assert response.status_code == expected_status
    assert calls == ["primary"]

def test_fallback_failure_stops_after_two_attempts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failing_provider(name: str):
        def execute(messages: list[main.Message]) -> str:
            calls.append(name)
            raise main.ProviderExecutionError(
                main.ProviderFailureKind.UNAVAILABLE,
                "Provider unavailable",
            )

        return execute

    monkeypatch.setitem(
        main.PROVIDERS, "provider-a", failing_provider("primary")
    )
    monkeypatch.setitem(
        main.PROVIDERS, "provider-b", failing_provider("secondary")
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(),
    )

    assert response.status_code == 503
    assert calls == ["primary", "secondary"]

def test_malformed_primary_does_not_trigger_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def malformed_provider(messages: list[main.Message]) -> str:
        calls.append("primary")
        return "   "

    def secondary(messages: list[main.Message]) -> str:
        calls.append("secondary")
        return "Should not execute"

    monkeypatch.setitem(
        main.PROVIDERS, "provider-a", malformed_provider
    )
    monkeypatch.setitem(main.PROVIDERS, "provider-b", secondary)

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(),
    )

    assert response.status_code == 502
    assert calls == ["primary"]

def test_missing_fallback_configuration_fails_before_execution(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def primary(messages: list[main.Message]) -> str:
        calls.append("primary")
        return "Success"

    monkeypatch.setitem(main.PROVIDERS, "provider-a", primary)
    monkeypatch.delitem(main.PROVIDERS, "provider-b")

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(),
    )

    assert response.status_code == 503
    assert calls == []

def test_programming_error_is_not_hidden(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def broken_provider(messages: list[main.Message]) -> str:
        calls.append("primary")
        raise RuntimeError("Internal programming defect")

    def secondary(messages: list[main.Message]) -> str:
        calls.append("secondary")
        return "Should not execute"

    monkeypatch.setitem(
        main.PROVIDERS, "provider-a", broken_provider
    )
    monkeypatch.setitem(main.PROVIDERS, "provider-b", secondary)

    with pytest.raises(
        RuntimeError,
        match="Internal programming defect",
    ):
        client.post(
            "/v1/chat/completions",
            json=request_payload(),
        )

    assert calls == ["primary"]

def test_existing_direct_route_has_no_implicit_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def primary(messages: list[main.Message]) -> str:
        calls.append("primary")
        raise main.ProviderExecutionError(
            main.ProviderFailureKind.UNAVAILABLE,
            "Unavailable",
        )

    def secondary(messages: list[main.Message]) -> str:
        calls.append("secondary")
        return "Should not execute"

    monkeypatch.setitem(main.PROVIDERS, "provider-a", primary)
    monkeypatch.setitem(main.PROVIDERS, "provider-b", secondary)

    response = client.post(
        "/v1/chat/completions",
        json=request_payload(model="mock-model"),
    )

    assert response.status_code == 503
    assert calls == ["primary"]
