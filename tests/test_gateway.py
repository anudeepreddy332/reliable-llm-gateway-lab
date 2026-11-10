from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

def payload(model="mock-model", content="Hello"):
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
    }

def test_provider_a_routing():
    response = client.post(
        "/v1/chat/completions",
        json=payload("mock-model", "Hello A"),
    )
    assert response.status_code == 200

    body = response.json()
    assert body["model"] == "mock-model"
    assert body["choices"][0]["message"]["content"] == (
        "Mock A received: Hello A"
    )

def test_provider_b_routing():
    response = client.post(
        "/v1/chat/completions",
        json=payload("mock-model-b", "Hello B"),
    )
    assert response.status_code == 200

    body = response.json()
    assert body["model"] == "mock-model-b"
    assert body["choices"][0]["message"]["content"] == (
        "Mock B received: Hello B"
    )

def test_selected_provider_receives_messages(monkeypatch):
    received = []

    def provider_spy(messages):
        received.extend(messages)
        return "Spy response"

    monkeypatch.setitem(
        main.PROVIDERS, "provider-a", provider_spy
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Spy response"
    )
    assert [(m.role, m.content) for m in received] == [
        ("system", "Be concise"),
        ("user", "Hello"),
    ]

def test_unknown_model():
    response = client.post(
        "/v1/chat/completions",
        json=payload("unknown-model"),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"

def test_missing_provider_configuration(monkeypatch):
    monkeypatch.setitem(
        main.MODEL_ROUTES, "mock-model", "missing-provider"
    )

    response = client.post(
        "/v1/chat/completions",
        json=payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Configured provider unavailable"
    )

def test_empty_messages():
    body = payload()
    body["messages"] = []

    response = client.post(
        "/v1/chat/completions", json=body
    )

    assert response.status_code == 422


def test_streaming_rejected():
    body = payload()
    body["stream"] = True

    response = client.post(
        "/v1/chat/completions", json=body
    )

    assert response.status_code == 422

def test_invalid_role():
    body = payload()
    body["messages"][0]["role"] = "developer"

    response = client.post(
        "/v1/chat/completions", json=body
    )

    assert response.status_code == 422


def test_empty_message_content():
    response = client.post(
        "/v1/chat/completions",
        json=payload(content=""),
    )

    assert response.status_code == 422