import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nexus_brain.api import create_app  # noqa: E402
from nexus_brain.brain import Brain  # noqa: E402


@pytest.fixture
def client():
    brain = Brain()
    brain.config.learning.reflect_every_n_cycles = 10_000
    return TestClient(create_app(brain))


def test_chat_and_status(client):
    r = client.post("/chat", json={"text": "calculate 6 * 7"})
    assert r.status_code == 200
    body = r.json()
    assert "42" in body["response"] and body["trace"] and body["mode"] == "system2"
    assert client.get("/status").json()["cycles"] == 1


def test_approval_flow_over_http(client):
    r = client.post("/chat", json={"text": "pay $20 to Bob"}).json()
    ticket = r["pending_approvals"][0]
    assert client.get("/approvals").json()[0]["id"] == ticket
    r2 = client.post(f"/approvals/{ticket}", json={"approve": True, "by": "alice"}).json()
    assert "completed" in r2["response"].lower()


def test_memory_audit_and_upgrades_endpoints(client):
    client.post("/chat", json={"text": "remember that my name is Ana"})
    mem = client.get("/memory", params={"kind": "semantic"}).json()
    assert any("Ana" in i["content"] for i in mem["items"])
    audit = client.get("/audit").json()
    assert audit["valid"] and audit["entries"]
    up = client.get("/upgrades").json()
    assert up["versions"][0]["label"] == "baseline"
    assert client.post("/upgrades/rollback", json={"reviewer": "alice"}).status_code == 409


def test_console_html(client):
    assert "Nexus Brain" in client.get("/").text
