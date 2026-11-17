import json

from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import main

@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(main.app) as test_client:
        yield test_client

def gateway_request() -> dict[str, object]:
    return {
        "model": "deepseek-flash",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
    }

def upstream_completion(
    content: str = "Hello from DeepSeek",
    finish_reason: str = "stop",
    model: str = "deepseek-flash",
) -> dict[str, object]:
    return {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1790000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    monkeypatch.setenv("GATEWAY_ENABLE_LIVE_PROVIDER", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")

    def create_mock_client() -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=5.0,
        )

    monkeypatch.setattr(
        main.deepseek,
        "create_client",
        create_mock_client,
    )

def test_live_access_disabled(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "GATEWAY_ENABLE_LIVE_PROVIDER",
        raising=False,
    )

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Live provider access disabled"
    )

def test_missing_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_ENABLE_LIVE_PROVIDER", "1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "DeepSeek provider not configured"
    )


def test_successful_provider_request(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=upstream_completion(),
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Hello from DeepSeek"
    )

    assert len(requests) == 1

    outbound = requests[0]
    assert outbound.method == "POST"
    assert str(outbound.url) == main.deepseek.DEEPSEEK_URL
    assert outbound.headers["authorization"] == (
        "Bearer test-only-key"
    )

    payload = json.loads(outbound.content)
    assert payload["model"] == "deepseek-flash"
    assert payload["messages"] == [
        {"role": "user", "content": "Hello"}
    ]
    assert payload["stream"] is False
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 96


@pytest.mark.parametrize(
    ("upstream_status", "gateway_status"),
    [
        (429, 429),
        (503, 503),
        (504, 504),
        (401, 502),
        (400, 502),
    ],
)
def test_upstream_http_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    upstream_status: int,
    gateway_status: int,
) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(
            upstream_status,
            json={"error": "PRIVATE_UPSTREAM_DIAGNOSTIC"},
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == gateway_status
    assert "PRIVATE_UPSTREAM_DIAGNOSTIC" not in response.text
    assert len(attempts) == 1

@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (httpx.ReadTimeout, 504),
        (httpx.ConnectError, 502),
    ],
)
def test_transport_failures(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[httpx.RequestError],
    expected_status: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure("Simulated transport failure", request=request)

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == expected_status

@pytest.mark.parametrize(
    "invalid_response",
    [
        {"choices": []},
        upstream_completion(finish_reason="length"),
        upstream_completion(model="unexpected-model"),
        upstream_completion(content="   "),
    ],
    ids=[
        "missing-choices",
        "truncated-completion",
        "incorrect-model",
        "blank-content",
    ],
)
def test_invalid_upstream_completion(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=invalid_response)

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 502

def test_invalid_upstream_json(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-valid-json",
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 502


def test_completion_whitespace_preserved(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "\n  Formatted response  \n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=upstream_completion(content=expected),
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"]
        == expected
    )