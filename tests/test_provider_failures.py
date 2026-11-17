from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import main

@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(main.app) as test_client:
        yield test_client

def valid_request() -> dict[str, object]:
    return {
        "model": "mock-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
    }

@pytest.mark.parametrize(
    ("failure_kind", "expected_status", "expected_detail"),
    [
        (
            main.ProviderFailureKind.RATE_LIMIT,
            429,
            "Upstream provider rate limited",
        ),
        (
            main.ProviderFailureKind.UNAVAILABLE,
            503,
            "Upstream provider unavailable",
        ),
        (
            main.ProviderFailureKind.TIMEOUT,
            504,
            "Upstream provider timed out",
        ),
        (
            main.ProviderFailureKind.CONNECTION,
            502,
            "Upstream connection failed",
        ),
    ],
)
def test_classified_provider_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: main.ProviderFailureKind,
    expected_status: int,
    expected_detail: str,
) -> None:
    calls: list[int] = []

    def failing_provider(
        messages: list[main.Message],
    ) -> str:
        calls.append(1)
        raise main.ProviderExecutionError(
            failure_kind,
            "Private upstream diagnostic: SECRET",
        )

    monkeypatch.setitem(
        main.PROVIDERS,
        "provider-a",
        failing_provider,
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request(),
    )

    assert response.status_code == expected_status
    assert response.json() == {
        "detail": expected_detail,
    }
    assert "SECRET" not in response.text
    assert len(calls) == 1

def test_unexpected_programming_error_propagates(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_provider(
        messages: list[main.Message],
    ) -> str:
        raise RuntimeError("Unexpected internal defect")

    monkeypatch.setitem(
        main.PROVIDERS,
        "provider-a",
        broken_provider,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected internal defect",
    ):
        client.post(
            "/v1/chat/completions",
            json=valid_request(),
        )


def test_valid_whitespace_is_preserved(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_content = "  Hello\n  World  \n"

    def formatted_provider(
        messages: list[main.Message],
    ) -> str:
        return expected_content

    monkeypatch.setitem(
        main.PROVIDERS,
        "provider-a",
        formatted_provider,
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request(),
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == expected_content
    )