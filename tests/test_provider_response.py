
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


def test_valid_completion_response(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=valid_request(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-model"
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Mock A received: Hello",
    }
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "invalid_output",
    [
        None,
        "",
        "   ",
        123,
        {"content": "unexpected object"},
    ],
    ids=[
        "none",
        "empty-string",
        "whitespace-only",
        "number",
        "object",
    ],
)
def test_invalid_provider_output_returns_502(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    invalid_output: object,
) -> None:

    def broken_provider(
        messages: list[main.Message],
    ) -> object:
        return invalid_output

    monkeypatch.setitem(
        main.PROVIDERS,
        "provider-a",
        broken_provider,
    )

    response = client.post(
        "/v1/chat/completions",
        json=valid_request(),
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Invalid provider response"
    }

    assert "unexpected object" not in response.text
