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
        "model": "cohere-command-r7b",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
    }

def upstream_completion(
    *,
    blocks: list[dict[str, str]] | None = None,
    finish_reason: str = "COMPLETE",
    role: str = "assistant",
) -> dict[str, object]:
    if blocks is None:
        blocks = [
            {"type": "text", "text": "Hello from Cohere"}
        ]

    return {
        "id": "cohere-test-123",
        "finish_reason": finish_reason,
        "message": {
            "role": role,
            "content": blocks,
        },
        "usage": {
            "billed_units": {
                "input_tokens": 5,
                "output_tokens": 4,
            },
            "tokens": {
                "input_tokens": 10,
                "output_tokens": 4,
            },
        },
    }

def install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    monkeypatch.setenv("GATEWAY_ENABLE_LIVE_PROVIDER", "1")
    monkeypatch.setenv("COHERE_API_KEY", "test-only-key")

    def create_mock_client() -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=5.0,
            trust_env=False,
        )

    monkeypatch.setattr(
        main.cohere,
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
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Cohere provider not configured"
    )

def test_successful_native_request(
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
    body = response.json()

    assert body["model"] == "cohere-command-r7b"
    assert body["choices"][0]["message"]["content"] == (
        "Hello from Cohere"
    )
    assert body["choices"][0]["finish_reason"] == "stop"

    assert len(requests) == 1

    outbound = requests[0]

    assert outbound.method == "POST"
    assert str(outbound.url) == main.cohere.COHERE_URL
    assert outbound.headers["authorization"] == (
        "Bearer test-only-key"
    )

    payload = json.loads(outbound.content)

    assert payload == {
        "model": "command-r7b-12-2024",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "max_tokens": 128,
    }


@pytest.mark.parametrize(
    ("upstream_status", "expected_status"),
    [
        (429, 429),
        (503, 503),
        (504, 504),
        (400, 502),
        (401, 502),
        (403, 502),
    ],
)
def test_upstream_http_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    upstream_status: int,
    expected_status: int,
) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(
            upstream_status,
            json={"message": "PRIVATE_UPSTREAM_ERROR"},
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == expected_status
    assert "PRIVATE_UPSTREAM_ERROR" not in response.text
    assert len(attempts) == 1


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (httpx.ReadTimeout, 504),
        (httpx.ConnectError, 502),
    ],
)
def test_transport_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[httpx.RequestError],
    expected_status: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure(
            "Simulated transport error",
            request=request,
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == expected_status

@pytest.mark.parametrize(
    "invalid_response",
    [
        upstream_completion(blocks=[]),
        upstream_completion(finish_reason="MAX_TOKENS"),
        upstream_completion(
            blocks=[{"type": "tool_call", "text": "action"}]
        ),
        upstream_completion(
            blocks=[{"type": "text", "text": "   "}]
        ),
        upstream_completion(role="user"),
        {
            "id": "test-id",
            "finish_reason": "COMPLETE",
        },
    ],
    ids=[
        "empty-content-blocks",
        "truncated-completion",
        "unsupported-content-block",
        "whitespace-only",
        "incorrect-role",
        "missing-message",
    ],
)
def test_invalid_upstream_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=invalid_response,
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Invalid provider response"
    )

def test_invalid_json(
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


def test_multiple_text_blocks(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=upstream_completion(
                blocks=[
                    {"type": "text", "text": "First "},
                    {"type": "text", "text": "second"},
                ]
            ),
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "First second"
    )


def test_whitespace_preserved(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "\n  Formatted output  \n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=upstream_completion(
                blocks=[{"type": "text", "text": expected}]
            ),
        )

    install_mock_transport(monkeypatch, handler)

    response = client.post(
        "/v1/chat/completions",
        json=gateway_request(),
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        expected
    )