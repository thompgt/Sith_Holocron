import json
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from src.llm.gemini_wrapper import GeminiChatWrapper
from src.llm.persona_manager import PersonaManager
from src.observability import metrics
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStoreManager

# --- configuration ------------------------------------------------------

#: Comma-separated list of browser origins allowed to call this API. Defaults to
#: the Vite dev server this repo's frontend runs on.
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5180,http://127.0.0.1:5180"

#: Longest accepted chat message. Every request is a paid Gemini call, so an
#: unbounded string is someone else's bill.
MAX_MESSAGE_CHARS = 2000

#: Per-IP fixed-window rate limit on the chat endpoint.
RATE_LIMIT_REQUESTS = int(os.getenv("HOLOCRON_RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("HOLOCRON_RATE_LIMIT_WINDOW", "60"))


def allowed_origins() -> list[str]:
    raw = os.getenv("HOLOCRON_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# --- rate limiting ------------------------------------------------------


class RateLimiter:
    """Fixed-window per-key request limiter.

    In-process and therefore per-worker: with N uvicorn workers the effective
    limit is N times the configured one. It is a guard against a single client
    draining the Gemini quota, not a distributed quota system -- that would need
    shared state.
    """

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, key: str, now: float | None = None) -> bool:
        """Record a request. Returns False when the caller is over its limit."""
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            window_start, count = self._windows.get(key, (now, 0))
            if now - window_start >= self.window_seconds:
                window_start, count = now, 0
            if count >= self.limit:
                return False
            self._windows[key] = (window_start, count + 1)

            # Opportunistic sweep so the dict cannot grow without bound from a
            # rotating set of source addresses.
            if len(self._windows) > 4096:
                self._windows = {
                    k: v
                    for k, v in self._windows.items()
                    if now - v[0] < self.window_seconds
                }
            return True


rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


# --- RAG engine ---------------------------------------------------------


@dataclass
class Engine:
    """The RAG components, assembled once at startup."""

    pm: PersonaManager
    retriever: HybridRetriever
    llm: GeminiChatWrapper
    vs_manager: VectorStoreManager


#: Populated by the lifespan handler. None until startup succeeds, which is what
#: /healthz reports on.
engine: Engine | None = None


def build_engine() -> Engine:
    """Construct the RAG engine, refusing to come up without an index.

    load() used to be called and its result thrown away. With the gitignored
    vector store absent that left search() returning nothing, so every answer was
    ungrounded and still HTTP 200 -- a fully broken deployment with no symptom
    anywhere. An index is a hard prerequisite, so failing to find one is a
    startup failure.
    """
    pm = PersonaManager()
    vs_manager = VectorStoreManager(persist_directory="data/vector_store")

    if not vs_manager.load():
        raise RuntimeError(
            "No vector index at data/vector_store/. The API only loads an "
            "existing index; build one first with `python -m src.main`."
        )

    return Engine(
        pm=pm,
        retriever=HybridRetriever(vs_manager, persona_manager=pm),
        llm=GeminiChatWrapper(),
        vs_manager=vs_manager,
    )


def require_engine() -> Engine:
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Holocron is not initialised.",
        )
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A module-level singleton assigned from the lifespan handler is the shape
    # FastAPI's startup hook has; the alternative (app.state) is the same global
    # with extra indirection. require_engine() is the only reader.
    global engine  # noqa: PLW0603
    engine = build_engine()

    # Startup facts. Recorded once so /metrics answers "is the index even
    # loaded?" before any traffic arrives.
    metrics.record_index_stats(engine.vs_manager)
    metrics.record_persona_stats(engine.pm)

    yield

    engine = None


app = FastAPI(lifespan=lifespan)

# Adds http_requests_total / http_request_duration_seconds for every route and
# serves the whole default registry (these plus the holocron_* domain metrics)
# at GET /metrics. Must run before the app starts serving so the route exists.
Instrumentator().instrument(app).expose(app)

# No cookies or Authorization headers are used, so credentials are not enabled.
# allow_origins=["*"] with allow_credentials=True was doubly wrong: browsers
# reject that combination outright, and a wildcard is not the intent past
# localhost anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    persona: str = Field(default="VADER", max_length=64)


@app.get("/healthz")
async def healthz():
    """Readiness, distinct from liveness.

    503 until the index is loaded, so an orchestrator does not route traffic to a
    process that would answer every question ungrounded.
    """
    if engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "index_loaded": False},
        )
    return {
        "status": "ok",
        "index_loaded": True,
        "personas": len(engine.pm.personas),
    }


@app.get("/api/personas")
async def get_personas():
    active = require_engine()
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in active.pm.personas.items()
    ]


@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    active = require_engine()

    client_key = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.check(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. The dark side rewards patience.",
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS))},
        )

    # Bounded label value; the raw request.persona is caller-controlled.
    persona_label = metrics.normalize_persona(request.persona)
    # Time-to-first-token is measured from here, not from the first generator
    # tick, so it includes retrieval and prompt assembly — the whole wait the
    # user actually experiences before text appears.
    request_start = time.perf_counter()

    try:
        docs = active.retriever.retrieve(request.message, character=request.persona, k=4)
        context = active.pm.format_context(docs)
        system_prompt = active.pm.get_system_prompt(request.persona)
    except Exception as e:
        # This half runs before StreamingResponse, so a failure here is a real
        # HTTP 500 and the instrumentator will see it. Counted anyway to keep
        # holocron_chat_* a complete account of chat outcomes.
        metrics.CHAT_ERRORS.labels(
            persona=persona_label, phase="retrieval", error_type=type(e).__name__
        ).inc()
        metrics.CHAT_REQUESTS.labels(persona=persona_label, outcome="error").inc()
        metrics.CHAT_STREAM_DURATION.labels(
            persona=persona_label, outcome="error"
        ).observe(time.perf_counter() - request_start)
        raise

    async def event_generator():
        first_token_seen = False
        completed = False
        try:
            # llm.stream_chat is a synchronous generator, and each next() blocks
            # on the network. Iterating it directly inside an async def would
            # block the event loop for the whole response: a second caller would
            # see nothing until the first finished, and /metrics scrapes would
            # stall behind them. iterate_in_threadpool moves each next() to a
            # worker thread so only that thread waits.
            stream = iterate_in_threadpool(
                active.llm.stream_chat(system_prompt, request.message, context)
            )
            async for chunk in stream:
                if chunk:
                    if not first_token_seen:
                        first_token_seen = True
                        metrics.CHAT_TIME_TO_FIRST_TOKEN.labels(
                            persona=persona_label
                        ).observe(time.perf_counter() - request_start)
                    metrics.CHAT_STREAMED_FRAMES.labels(persona=persona_label).inc()
                    # SSE format: data: <content>\n\n
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            completed = True
            metrics.CHAT_REQUESTS.labels(persona=persona_label, outcome="success").inc()
            metrics.CHAT_STREAM_DURATION.labels(
                persona=persona_label, outcome="success"
            ).observe(time.perf_counter() - request_start)
            yield "data: [DONE]\n\n"
        except Exception as e:
            # THE ONLY PLACE A CHAT FAILURE IS OBSERVABLE.
            #
            # This except turns the exception into a 200-status SSE error frame.
            # By the time it runs the response headers are long gone, so no
            # FastAPI exception handler, no middleware, and nothing in
            # http_requests_total will ever record this as a failure — the
            # request looks like a clean 200 from the outside. A dead
            # GOOGLE_API_KEY or a Gemini outage would show up as perfectly
            # healthy traffic. Incrementing here is what makes the failure
            # visible, so keep the counter inside this block.
            completed = True
            metrics.CHAT_ERRORS.labels(
                persona=persona_label, phase="llm_stream", error_type=type(e).__name__
            ).inc()
            metrics.CHAT_REQUESTS.labels(persona=persona_label, outcome="error").inc()
            metrics.CHAT_STREAM_DURATION.labels(
                persona=persona_label, outcome="error"
            ).observe(time.perf_counter() - request_start)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Starlette throws GeneratorExit/CancelledError into the generator
            # when the client hangs up mid-stream. Neither reaches the except
            # above (GeneratorExit is a BaseException), so the request would
            # otherwise be counted nowhere at all.
            if not completed:
                metrics.CHAT_CLIENT_DISCONNECTS.labels(persona=persona_label).inc()
                metrics.CHAT_REQUESTS.labels(
                    persona=persona_label, outcome="disconnected"
                ).inc()
                metrics.CHAT_STREAM_DURATION.labels(
                    persona=persona_label, outcome="disconnected"
                ).observe(time.perf_counter() - request_start)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
