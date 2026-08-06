import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from prometheus_fastapi_instrumentator import Instrumentator

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.gemini_wrapper import GeminiChatWrapper
from src.llm.persona_manager import PersonaManager
from src.observability import metrics

app = FastAPI()

# Adds http_requests_total / http_request_duration_seconds for every route and
# serves the whole default registry (these plus the holocron_* domain metrics)
# at GET /metrics. Must run before the app starts serving so the route exists.
Instrumentator().instrument(app).expose(app)

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
pm = PersonaManager()
vs_manager = VectorStoreManager(persist_directory="data/vector_store")
vs_manager.load()
retriever = HybridRetriever(vs_manager)
llm = GeminiChatWrapper()

# Startup facts. Recorded once at import so /metrics answers "is the index even
# loaded?" before any traffic arrives — an empty index yields ungrounded answers
# with no HTTP-level symptom at all.
metrics.record_index_stats(vs_manager)
metrics.record_persona_stats(pm)

class ChatRequest(BaseModel):
    message: str
    persona: str = "VADER"

@app.get("/api/personas")
async def get_personas():
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in pm.personas.items()
    ]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Bounded label value; the raw request.persona is caller-controlled.
    persona_label = metrics.normalize_persona(request.persona)
    # Time-to-first-token is measured from here, not from the first generator
    # tick, so it includes retrieval and prompt assembly — the whole wait the
    # user actually experiences before text appears.
    request_start = time.perf_counter()

    try:
        docs = retriever.retrieve(request.message, character=request.persona, k=4)
        context = pm.format_context(docs)
        system_prompt = pm.get_system_prompt(request.persona)
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
            for chunk in llm.stream_chat(system_prompt, request.message, context):
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
