"""Tests for the FastAPI surface and its SSE outcome bookkeeping.

The chat endpoint's failure accounting is the subtle part of this codebase and
had no test at all. Once an SSE response has sent headers, an exception cannot
become an HTTP 5xx -- the request stays a clean 200 forever -- so the only record
that anything went wrong is the counter incremented inside the generator. This
file pins all three outcome labels, including the disconnect path that rides on
GeneratorExit being a BaseException and therefore missing the `except` above it.
"""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import src.api.server as server
from src.llm.persona_manager import PersonaManager
from src.observability import metrics


# --- stubs --------------------------------------------------------------


class StubRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            Document(page_content="Lore about the Sith.", metadata={"type": "lore"})
        ]

    def retrieve(self, query, character=None, k=4):
        return self.docs


class RaisingRetriever:
    def retrieve(self, query, character=None, k=4):
        raise RuntimeError("index exploded")


class StubLLM:
    """Synchronous streaming generator, matching GeminiChatWrapper.stream_chat."""

    def __init__(self, chunks=("I ", "am ", "your father."), raise_after=None):
        self.chunks = chunks
        self.raise_after = raise_after
        self.calls = []

    def stream_chat(self, system_prompt, message, context):
        self.calls.append((system_prompt, message, context))
        for i, chunk in enumerate(self.chunks):
            if self.raise_after is not None and i == self.raise_after:
                raise RuntimeError("gemini is down")
            yield chunk


def make_engine(retriever=None, llm=None):
    return server.Engine(
        pm=PersonaManager(),
        retriever=retriever if retriever is not None else StubRetriever(),
        llm=llm if llm is not None else StubLLM(),
        vs_manager=None,
    )


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Each test gets a fresh window; the limiter is module-level state."""
    server.rate_limiter = server.RateLimiter(
        server.RATE_LIMIT_REQUESTS, server.RATE_LIMIT_WINDOW_SECONDS
    )
    yield


@pytest.fixture
def client_factory(monkeypatch):
    """A TestClient whose lifespan installs stubs instead of the real engine."""

    def factory(retriever=None, llm=None):
        engine = make_engine(retriever, llm)
        monkeypatch.setattr(server, "build_engine", lambda: engine)
        # record_index_stats expects a real VectorStoreManager.
        monkeypatch.setattr(metrics, "record_index_stats", lambda vs: None)
        return TestClient(server.app), engine

    return factory


def counter_value(counter, **labels):
    """Current value of a labelled counter, 0.0 when the child does not exist."""
    return counter.labels(**labels)._value.get()


def sse_frames(text):
    return [
        line[len("data: "):]
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


# --- startup and readiness ---------------------------------------------


def test_healthz_is_503_before_startup():
    # No lifespan run, so the engine was never built.
    assert server.engine is None
    response = TestClient(server.app).get("/healthz")
    assert response.status_code == 503
    assert response.json()["index_loaded"] is False


def test_healthz_is_ok_once_the_engine_is_up(client_factory):
    client, _ = client_factory()
    with client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["index_loaded"] is True


def test_build_engine_refuses_to_start_without_an_index(monkeypatch, tmp_path):
    # The original bug: load()'s False was discarded, so the app served
    # ungrounded answers with HTTP 200 forever.
    class EmptyManager:
        def __init__(self, *a, **kw):
            pass

        def load(self):
            return False

    monkeypatch.setattr(server, "VectorStoreManager", EmptyManager)
    with pytest.raises(RuntimeError, match="No vector index"):
        server.build_engine()


def test_endpoints_are_503_when_the_engine_is_absent():
    client = TestClient(server.app)
    assert client.get("/api/personas").status_code == 503
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 503


# --- SSE outcome bookkeeping ------------------------------------------


def test_successful_stream_frames_and_counts_success(client_factory):
    client, _ = client_factory(llm=StubLLM(chunks=("I ", "am ", "your father.")))
    before = counter_value(metrics.CHAT_REQUESTS, persona="VADER", outcome="success")

    with client:
        response = client.post("/api/chat", json={"message": "who are you?"})

    assert response.status_code == 200
    frames = sse_frames(response.text)
    assert frames[-1] == "[DONE]"
    assert [json.loads(f)["content"] for f in frames[:-1]] == [
        "I ", "am ", "your father.",
    ]

    after = counter_value(metrics.CHAT_REQUESTS, persona="VADER", outcome="success")
    assert after == before + 1


def test_midstream_failure_counts_error_despite_http_200(client_factory):
    client, _ = client_factory(llm=StubLLM(chunks=("a", "b", "c"), raise_after=1))
    before = counter_value(metrics.CHAT_REQUESTS, persona="VADER", outcome="error")
    errors_before = counter_value(
        metrics.CHAT_ERRORS,
        persona="VADER",
        phase="llm_stream",
        error_type="RuntimeError",
    )

    with client:
        response = client.post("/api/chat", json={"message": "hi"})

    # The whole point: a failed generation is still a 200.
    assert response.status_code == 200
    frames = sse_frames(response.text)
    assert "[DONE]" not in frames
    assert json.loads(frames[-1])["error"] == "gemini is down"

    assert counter_value(
        metrics.CHAT_REQUESTS, persona="VADER", outcome="error"
    ) == before + 1
    assert counter_value(
        metrics.CHAT_ERRORS,
        persona="VADER",
        phase="llm_stream",
        error_type="RuntimeError",
    ) == errors_before + 1


def test_client_disconnect_counts_disconnected(monkeypatch):
    # GeneratorExit is a BaseException, so it bypasses the `except Exception`
    # branch entirely; only the `finally` sees this outcome.
    #
    # Driven through the response's body_iterator rather than TestClient, which
    # drains a streaming response to completion and so can only ever produce the
    # success path. aclose() on a partially consumed async generator is precisely
    # what Starlette does when a client hangs up: it throws GeneratorExit in at
    # the suspended yield.
    monkeypatch.setattr(server, "engine", make_engine(llm=StubLLM(chunks=tuple("abcdefghij"))))
    before = counter_value(
        metrics.CHAT_REQUESTS, persona="VADER", outcome="disconnected"
    )
    disconnects_before = counter_value(
        metrics.CHAT_CLIENT_DISCONNECTS, persona="VADER"
    )

    class FakeClient:
        host = "test-client"

    class FakeRequest:
        client = FakeClient()

    async def consume_one_frame_then_hang_up():
        response = await server.chat(server.ChatRequest(message="hi"), FakeRequest())
        iterator = response.body_iterator
        first = await iterator.__anext__()
        assert json.loads(first[len("data: "):])["content"] == "a"
        await iterator.aclose()

    asyncio.run(consume_one_frame_then_hang_up())

    assert counter_value(
        metrics.CHAT_REQUESTS, persona="VADER", outcome="disconnected"
    ) == before + 1
    assert counter_value(
        metrics.CHAT_CLIENT_DISCONNECTS, persona="VADER"
    ) == disconnects_before + 1


class BlockingLLM:
    """Streams with a genuinely blocking wait per chunk, like a network call."""

    def __init__(self, chunks=3, delay=0.15):
        self.chunks = chunks
        self.delay = delay

    def stream_chat(self, system_prompt, message, context):
        for i in range(self.chunks):
            time.sleep(self.delay)
            yield f"chunk{i}"


def test_concurrent_streams_do_not_serialize(monkeypatch):
    """Two callers must stream at the same time.

    stream_chat is a synchronous generator whose every next() blocks. Iterated
    directly inside an `async def` it would hold the event loop for the entire
    response: a second caller would see nothing until the first finished, and
    /metrics scrapes would queue behind them. This asserts the wall clock for two
    concurrent requests is far below their combined serial cost.
    """
    llm = BlockingLLM(chunks=3, delay=0.15)
    serial_cost = 3 * 0.15 * 2
    monkeypatch.setattr(server, "engine", make_engine(llm=llm))
    monkeypatch.setattr(server, "rate_limiter", server.RateLimiter(0, 60.0))

    async def two_at_once():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            started = time.monotonic()
            responses = await asyncio.gather(
                client.post("/api/chat", json={"message": "one"}),
                client.post("/api/chat", json={"message": "two"}),
            )
            return time.monotonic() - started, responses

    elapsed, responses = asyncio.run(two_at_once())

    for response in responses:
        assert response.status_code == 200
        assert sse_frames(response.text)[-1] == "[DONE]"

    # Overlapped: ~0.45s rather than ~0.90s. The bound is generous so the test
    # fails on serialization, not on a slow machine.
    assert elapsed < serial_cost * 0.75, (
        f"streams serialized: {elapsed:.2f}s for two requests whose serial cost "
        f"is {serial_cost:.2f}s"
    )


def test_retrieval_failure_is_a_real_500(client_factory):
    # This half runs before the response starts, so it can still be a 5xx.
    client, _ = client_factory(retriever=RaisingRetriever())
    before = counter_value(
        metrics.CHAT_ERRORS,
        persona="VADER",
        phase="retrieval",
        error_type="RuntimeError",
    )

    with client:
        with pytest.raises(RuntimeError, match="index exploded"):
            client.post("/api/chat", json={"message": "hi"})

    assert counter_value(
        metrics.CHAT_ERRORS,
        persona="VADER",
        phase="retrieval",
        error_type="RuntimeError",
    ) == before + 1


def test_unknown_persona_is_normalized_in_labels(client_factory):
    client, _ = client_factory()
    with client:
        response = client.post(
            "/api/chat", json={"message": "hi", "persona": "NOT_A_PERSONA"}
        )
    assert response.status_code == 200
    # An arbitrary persona string must not mint a new time series.
    assert metrics.normalize_persona("NOT_A_PERSONA") != "NOT_A_PERSONA"


# --- input bounds and rate limiting -----------------------------------


def test_overlong_message_is_rejected(client_factory):
    client, _ = client_factory()
    with client:
        response = client.post(
            "/api/chat", json={"message": "x" * (server.MAX_MESSAGE_CHARS + 1)}
        )
    assert response.status_code == 422


def test_empty_message_is_rejected(client_factory):
    client, _ = client_factory()
    with client:
        assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_rate_limit_returns_429_with_retry_after(client_factory, monkeypatch):
    client, _ = client_factory()
    monkeypatch.setattr(server, "rate_limiter", server.RateLimiter(2, 60.0))

    with client:
        assert client.post("/api/chat", json={"message": "1"}).status_code == 200
        assert client.post("/api/chat", json={"message": "2"}).status_code == 200
        third = client.post("/api/chat", json={"message": "3"})

    assert third.status_code == 429
    assert third.headers["Retry-After"] == "60"


# --- rate limiter unit behaviour ---------------------------------------


def test_rate_limiter_allows_up_to_the_limit():
    limiter = server.RateLimiter(3, 60.0)
    assert [limiter.check("ip", now=0.0) for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_window_rolls_over():
    limiter = server.RateLimiter(1, 10.0)
    assert limiter.check("ip", now=0.0) is True
    assert limiter.check("ip", now=5.0) is False
    assert limiter.check("ip", now=10.0) is True


def test_rate_limiter_keys_are_independent():
    limiter = server.RateLimiter(1, 60.0)
    assert limiter.check("a", now=0.0) is True
    assert limiter.check("b", now=0.0) is True
    assert limiter.check("a", now=0.0) is False


def test_rate_limit_of_zero_disables_the_limiter():
    limiter = server.RateLimiter(0, 60.0)
    assert all(limiter.check("ip", now=0.0) for _ in range(100))


# --- CORS ---------------------------------------------------------------


def test_cors_is_not_a_wildcard_with_credentials(monkeypatch):
    monkeypatch.delenv("HOLOCRON_ALLOWED_ORIGINS", raising=False)
    origins = server.allowed_origins()
    assert "*" not in origins
    assert origins


def test_allowed_origins_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("HOLOCRON_ALLOWED_ORIGINS", "https://a.example, https://b.example")
    assert server.allowed_origins() == ["https://a.example", "https://b.example"]
