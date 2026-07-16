# Sith Holocron: Chronicle of the Sith

A Retrieval-Augmented Generation (RAG) chatbot that lets you converse with
Star Wars Sith characters (currently Darth Vader, Emperor Palpatine, and a
"Generic Sith Lord" persona) in their own voice. User queries are answered by
an LLM whose responses are grounded in a vector store of scraped Wookieepedia
lore and film-script dialogue, combined with a persona-specific system prompt
designed to keep the "Sith voice" consistent.

The project has both a Python CLI and a web app (FastAPI backend + React
frontend) built around the same RAG engine.

## How it works

1. **Ingestion**: Wookieepedia articles (`scripts/scrape_wookieepedia.py`)
   and prequel/original-trilogy script dialogue (`data/raw/prequel-csv/*.csv`,
   `data/raw/lore.json`) are parsed and chunked
   (`src/ingestion/lore_processor.py`, `src/ingestion/script_parser.py`).
2. **Indexing**: Chunks are embedded with a Hugging Face sentence-transformer
   (`sentence-transformers/all-MiniLM-L6-v2`) and stored in a local FAISS
   vector index (`src/retrieval/vector_store.py`).
3. **Retrieval**: `src/retrieval/hybrid_retriever.py` queries the vector
   store and blends factual "lore" chunks with character-specific "dialogue"
   chunks (filtered by the selected persona) to balance grounding with
   stylistic voice.
4. **Persona & prompting**: `src/llm/persona_manager.py` defines the
   available personas (Vader, Palpatine, Generic Sith) and builds a system
   prompt instructing the model to stay in character and use the retrieved
   context rather than hallucinate.
5. **Generation**: `src/llm/gemini_wrapper.py` sends the prompt + context to
   Google's Gemini (`gemini-1.5-flash` by default via `langchain-google-genai`)
   and streams the response back token-by-token.

Two front doors exist for this pipeline:

- **CLI** (`src/main.py`): a `rich`-powered terminal chat interface. Lets you
  pick a persona and chat with streaming output directly in the console.
- **Web app**: `src/api/server.py` is a FastAPI backend exposing
  `GET /api/personas` and a streaming `POST /api/chat` (Server-Sent Events).
  `frontend/` is a Vite + React + TypeScript + Tailwind CSS single-page app
  (with `framer-motion` animations and a CRT/terminal-styled UI) that talks
  to the backend at `http://localhost:8000`.

## Tech stack

- **Backend**: Python, LangChain (`langchain`, `langchain-community`,
  `langchain-google-genai`, `langchain-huggingface`), FAISS (`faiss-cpu`) for
  vector storage, `sentence-transformers` for embeddings, FastAPI + Uvicorn +
  `sse-starlette` for the streaming API, `rich` for the CLI, `pandas` for
  script/CSV parsing, `pytest` for tests.
- **LLM**: Google Gemini, via `GOOGLE_API_KEY`. (Note: `requirements.txt`
  also lists `langchain-openai`, but no OpenAI wrapper is currently wired up
  in `src/llm` — only the Gemini wrapper is used.)
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Framer Motion,
  Lucide icons.

## Project structure

```
PROTOCOL.md          Engineering/testing conventions for the project
WORKPLAN.md           Phased project plan and status checklist
data/
  raw/                Scraped/raw lore (Wookieepedia JSON) and script CSVs
  processed/           Pre-built datasets (dialogue JSON/JSONL)
scripts/
  scrape_wookieepedia.py   Scrapes Star Wars Fandom wiki pages into data/raw
  parse_scripts.py          Parses script/dialogue source files
  synthesize_dataset.py     Builds a processed training/eval dataset
src/
  ingestion/          Lore chunking (lore_processor.py) and dialogue
                       extraction (script_parser.py)
  retrieval/           FAISS vector store manager and hybrid retriever
  llm/                 Gemini chat wrapper and persona/prompt manager
  api/server.py        FastAPI backend (personas + streaming chat endpoints)
  main.py               Rich-based terminal chat client
tests/                 Pytest suite (unit tests per module + a persona
                       "voice consistency" audit)
frontend/              Vite + React + TypeScript web client
requirements.txt       Python dependencies
```

## Setup

### Backend (Python)

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set a Google API key for Gemini (the CLI and API server both read this
   from the environment, e.g. via a `.env` file loaded with `python-dotenv`):

   ```
   GOOGLE_API_KEY=your-key-here
   ```

3. Populate the raw data (optional — some processed data already exists
   under `data/processed/`):

   ```bash
   python scripts/scrape_wookieepedia.py
   ```

   The vector index is built automatically on first run if
   `data/vector_store/` doesn't already exist yet.

### Frontend (React)

```bash
cd frontend
npm install
```

## Running the project

**CLI chat client:**

```bash
python -m src.main
```

Select a persona (Darth Vader, Emperor Palpatine, or Generic Sith Lord) and
chat in the terminal.

**Web app:**

```bash
# Terminal 1: backend API on http://localhost:8000
python -m src.api.server

# Terminal 2: frontend dev server
cd frontend
npm run dev
```

## Testing

```bash
pytest
```

The `tests/` directory includes unit tests for the lore processor, script
parser, vector store, hybrid retriever, persona manager, and Gemini wrapper,
plus a `persona_audit.py` check aimed at verifying that generated responses
stay in-character.

## Project status

Per `WORKPLAN.md`, this is an early/mid-stage project. The core RAG pipeline,
persona system, CLI, and a first pass at the web UI (React frontend +
FastAPI streaming backend) exist, but items like full "Persona Filter"
hybrid-search tuning, end-to-end stress testing, and a deployment guide are
still marked as outstanding in the workplan.
