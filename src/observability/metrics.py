"""
Prometheus metric families for the Sith Holocron RAG backend.

Everything registers into prometheus_client's default ``REGISTRY``, which is also
what ``prometheus_fastapi_instrumentator`` exposes at ``/metrics`` — so the HTTP
metrics it generates and the domain metrics defined here are served together.

Naming convention: ``holocron_<subsystem>_<unit>``. Three subsystems matter here,
and they answer different questions:

* ``retrieval_*`` — is the hybrid retriever actually finding anything, and is the
  lore/dialogue rebalance doing what ``lore_weight`` claims?
* ``llm_*`` / ``chat_*`` — is Gemini responding, how fast does the *first* token
  arrive, and how long does the whole stream take?
* ``index_*`` / ``personas_*`` — startup facts about the loaded FAISS index.

**Label cardinality.** The only free-form value anywhere near these metrics is the
``persona`` field on an inbound chat request, which is attacker-controlled. Every
metric that carries a persona label runs it through :func:`normalize_persona`
first, which collapses anything unrecognised to ``other``. That caps the label
space at four values no matter what is POSTed.
"""

from typing import Optional

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# Re-exported so callers have one obvious place to reach the registry from.
registry = REGISTRY

# Mirrors PersonaManager.personas. Kept as a literal rather than imported to avoid
# a circular import (persona_manager is imported by the API layer, which imports
# this module) and to keep the metric label space fixed even if personas grow.
KNOWN_PERSONAS = ("VADER", "PALPATINE", "GENERIC_SITH")
UNKNOWN_PERSONA = "other"


def normalize_persona(persona: Optional[str]) -> str:
    """
    Collapse a caller-supplied persona to a bounded label value.

    ``/api/chat`` accepts an arbitrary ``persona`` string and PersonaManager
    silently falls back to GENERIC_SITH for unknown values. Using the raw string
    as a metric label would let any client mint unbounded new time series, so
    anything off the known list becomes ``other``.
    """
    if not persona:
        return UNKNOWN_PERSONA
    key = persona.strip().upper()
    return key if key in KNOWN_PERSONAS else UNKNOWN_PERSONA


# --------------------------------------------------------------------------- #
# Retrieval — src/retrieval/hybrid_retriever.py
# --------------------------------------------------------------------------- #

# FAISS similarity search is local and CPU-bound, so the interesting range is
# milliseconds to a couple of seconds, not the tens of seconds an API call needs.
RETRIEVAL_DURATION = Histogram(
    "holocron_retrieval_duration_seconds",
    "Wall-clock time for one HybridRetriever.retrieve() call, including the "
    "underlying FAISS similarity search and the lore/dialogue rebalance.",
    ["persona"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# k defaults to 4 and the retriever never returns more, but it can return *fewer*
# when the index is empty or the persona filter drops every dialogue hit — which
# is exactly the failure this histogram is here to make visible.
RETRIEVAL_DOCUMENTS = Histogram(
    "holocron_retrieval_documents",
    "Number of documents returned by one retrieve() call. Below k means the "
    "index is empty or filtering removed candidates.",
    ["persona"],
    buckets=(0, 1, 2, 3, 4, 6, 8, 12),
)

# The point of the hybrid retriever: lore supplies facts, dialogue supplies voice.
# Splitting the counter by source_type lets a single ratio panel show whether the
# lore_weight rebalance is landing near its target or starving one pool.
RETRIEVAL_DOCUMENTS_BY_TYPE = Counter(
    "holocron_retrieval_documents_total",
    "Documents returned by retrieve(), split by corpus. The lore:dialogue ratio "
    "is the observable form of the lore_weight rebalance.",
    ["persona", "source_type"],
)

# Candidates come out of FAISS before filtering; comparing this to the returned
# total shows how much the character filter is discarding.
RETRIEVAL_CANDIDATES = Counter(
    "holocron_retrieval_candidates_total",
    "Raw candidates returned by the FAISS over-fetch (k*3) before the persona "
    "filter and rebalance are applied.",
    ["persona"],
)

RETRIEVAL_ERRORS = Counter(
    "holocron_retrieval_errors_total",
    "retrieve() calls that raised, by exception class.",
    ["persona", "error_type"],
)


# --------------------------------------------------------------------------- #
# LLM — src/llm/gemini_wrapper.py
# --------------------------------------------------------------------------- #

LLM_REQUESTS = Counter(
    "holocron_llm_requests_total",
    "Calls into GeminiChatWrapper, by method and outcome.",
    ["model", "method", "outcome"],
)

# Gemini's stream yields content chunks, not tokens; the SDK does not surface a
# token count on the streaming path. Chunks (and characters below) are the honest
# proxies available here — do not read them as billing tokens.
LLM_STREAM_CHUNKS = Counter(
    "holocron_llm_stream_chunks_total",
    "Content chunks yielded by stream_chat(). A proxy for output tokens: the "
    "streaming API does not report a token count.",
    ["model"],
)

LLM_RESPONSE_CHARACTERS = Counter(
    "holocron_llm_response_characters_total",
    "Total characters of generated text yielded by stream_chat().",
    ["model"],
)

LLM_PROMPT_CHARACTERS = Counter(
    "holocron_llm_prompt_characters_total",
    "Total characters sent to the model (system prompt + context + query).",
    ["model", "method"],
)

# Time-to-first-chunk measured at the wrapper. Separate from the endpoint-level
# TTFT below, which additionally includes retrieval and prompt assembly.
LLM_TIME_TO_FIRST_CHUNK = Histogram(
    "holocron_llm_time_to_first_chunk_seconds",
    "Time from the stream_chat() call until the model's first content chunk "
    "arrives — the model's own think time, excluding retrieval.",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0),
)

LLM_STREAM_DURATION = Histogram(
    "holocron_llm_stream_duration_seconds",
    "Total time to exhaust a stream_chat() generator, by outcome.",
    ["model", "outcome"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
)


# --------------------------------------------------------------------------- #
# Chat endpoint — src/api/server.py
# --------------------------------------------------------------------------- #

CHAT_REQUESTS = Counter(
    "holocron_chat_requests_total",
    "POST /api/chat requests by persona and final outcome.",
    ["persona", "outcome"],
)

# The reason this metric has to exist at all: event_generator() catches every
# exception and emits it as a `data: {"error": ...}` SSE frame with HTTP 200
# already on the wire. FastAPI exception handlers never see chat failures, and
# the instrumentator's http_requests_total records a 200. Without an explicit
# counter incremented inside that except block, a total Gemini outage is
# invisible in every HTTP-level metric.
CHAT_ERRORS = Counter(
    "holocron_chat_errors_total",
    "Chat failures caught inside the SSE generator and emitted as an error "
    "frame. These return HTTP 200, so they do NOT appear as 5xx in the HTTP "
    "metrics — this counter is the only signal for them.",
    ["persona", "phase", "error_type"],
)

# The two halves of streaming latency. For a token stream these move independently:
# TTFT is what the user perceives as responsiveness, total duration is dominated by
# answer length. Averaged into one number, a slow first token hides behind short
# answers and a long answer looks like a latency regression.
CHAT_TIME_TO_FIRST_TOKEN = Histogram(
    "holocron_chat_time_to_first_token_seconds",
    "Time from request arrival until the first SSE content frame is yielded. "
    "Includes retrieval and prompt assembly as well as model think time.",
    ["persona"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0),
)

CHAT_STREAM_DURATION = Histogram(
    "holocron_chat_stream_duration_seconds",
    "Time from request arrival until the SSE stream terminates, by outcome.",
    ["persona", "outcome"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
)

CHAT_STREAMED_FRAMES = Counter(
    "holocron_chat_streamed_frames_total",
    "SSE content frames written to clients.",
    ["persona"],
)

# Streams that ended without a [DONE] and without an error frame — the client hung
# up mid-generation. Counted separately because it is normal user behaviour, not a
# fault, but it skews stream-duration percentiles downward.
CHAT_CLIENT_DISCONNECTS = Counter(
    "holocron_chat_client_disconnects_total",
    "Streams abandoned by the client before completion.",
    ["persona"],
)


# --------------------------------------------------------------------------- #
# Startup facts — set once from the module-level singletons in server.py
# --------------------------------------------------------------------------- #

INDEX_LOADED = Gauge(
    "holocron_index_loaded",
    "1 if a FAISS index was successfully loaded from disk at startup, else 0. "
    "0 means every retrieval returns empty and answers are ungrounded.",
)

INDEX_VECTORS = Gauge(
    "holocron_index_vectors",
    "Vectors in the loaded FAISS index (index.ntotal).",
)

INDEX_DOCUMENTS = Gauge(
    "holocron_index_documents",
    "Documents in the FAISS docstore. Normally equal to the vector count.",
)

INDEX_DIMENSION = Gauge(
    "holocron_index_dimension",
    "Embedding dimensionality of the FAISS index (384 for all-MiniLM-L6-v2).",
)

PERSONAS_LOADED = Gauge(
    "holocron_personas_loaded",
    "Personas registered in PersonaManager and served by GET /api/personas.",
)


def record_index_stats(vs_manager) -> None:
    """
    Populate the startup gauges from a loaded VectorStoreManager.

    Reaches into FAISS internals (``index.ntotal``, ``docstore._dict``) which are
    not part of LangChain's public surface, so every read is defensive: a gauge
    stays at 0 rather than taking down app startup if the layout changes.
    """
    store = getattr(vs_manager, "vector_store", None)
    if store is None:
        INDEX_LOADED.set(0)
        INDEX_VECTORS.set(0)
        INDEX_DOCUMENTS.set(0)
        INDEX_DIMENSION.set(0)
        return

    INDEX_LOADED.set(1)

    faiss_index = getattr(store, "index", None)
    if faiss_index is not None:
        INDEX_VECTORS.set(getattr(faiss_index, "ntotal", 0) or 0)
        INDEX_DIMENSION.set(getattr(faiss_index, "d", 0) or 0)

    docstore = getattr(store, "docstore", None)
    inner = getattr(docstore, "_dict", None)
    if inner is not None:
        INDEX_DOCUMENTS.set(len(inner))


def record_persona_stats(persona_manager) -> None:
    """Populate the persona gauge from a PersonaManager."""
    PERSONAS_LOADED.set(len(getattr(persona_manager, "personas", {}) or {}))
