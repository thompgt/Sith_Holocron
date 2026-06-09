# Sith Holocron: Chronicle of the Sith (RAG) - Workplan

## Phase 1: Environment & Infrastructure
- [ ] **Task 1**: Initialize Python environment, install core dependencies (LangChain/LlamaIndex, FAISS/ChromaDB, OpenAI/Anthropic SDKs).
- [ ] **Task 2**: Establish project directory structure for the RAG pipeline (`src/ingestion`, `src/retrieval`, `src/llm`).

## Phase 2: Data Ingestion & Indexing
- [ ] **Task 3**: Implement the Ingestion Engine to clean and chunk Wookieepedia lore (JSON).
- [ ] **Task 4**: Implement the Script Parser to extract dialogue snippets with character metadata.
- [ ] **Task 5**: Develop the Vector Store manager to index processed data using embeddings.

## Phase 3: Retrieval & Context Management
- [ ] **Task 6**: Build the Retrieval Logic (Hybrid Search) to fetch both factual lore and stylistic dialogue.
- [ ] **Task 7**: Implement a "Persona Filter" to prioritize a specific character's voice in the retrieval results.

## Phase 4: LLM Integration & Prompt Engineering
- [ ] **Task 8**: Design the "Sith Lord" System Prompt template with dynamic context injection.
- [ ] **Task 9**: Create the Chat Completion wrapper to handle streaming and history management.

## Phase 5: Verification & UI
- [x] **Task 10**: Implement the "Persona Audit" test suite to verify tone and factual grounding.
- [x] **Task 11**: Develop a simple CLI or Streamlit interface for interacting with the Holocron.
- [ ] **Task 12**: Final E2E system stress test and documentation of "The Sith Way" (User Guide).

## Phase 6: Web Migration (React + FastAPI)
- [ ] **Task 13**: Implement FastAPI backend with streaming chat (SSE).
- [ ] **Task 14**: Bootstrap Vite + React frontend with custom CSS (Holocron theme).
- [ ] **Task 15**: Implement interactive chat UI with persona selection and CRT effects.
- [ ] **Task 16**: Final E2E web validation and deployment guide.
