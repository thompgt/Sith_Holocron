# Sith Holocron - Workplan

## Status Overview
- [x] Initial Research & Plan Design
- [x] Project Structure Setup
- [x] Data Acquisition (Wookieepedia & Movie Scripts)
- [ ] Data Processing & Dataset Construction (**In Progress**)
- [ ] Cloud Fine-Tuning Setup
- [ ] Backend API Development
- [ ] Frontend Interface Development

---

## Remaining Tasks

### Phase 1: Data Processing & Dataset Construction
- [x] **Run Parser**: Execute `scripts/parse_scripts.py` to extract character-specific dialogue pairs.
- [ ] **Data Synthesis**: Create a script to combine extracted dialogues with lore context to form a JSONL instruction dataset (ShareGPT or Instruct format).
- [ ] **Quality Review**: Manually inspect the dataset for persona consistency.

### Phase 2: Cloud Fine-Tuning Setup
- [ ] **Unsloth Notebook**: Develop the Jupyter notebook for Llama 3.2 QLoRA fine-tuning.
- [ ] **Persona Training**: Set up the training loops for the "Sith Lord" and "Protocol Droid" adapters.
- [ ] **Export Adapters**: Save the resulting LoRA weights for deployment.

### Phase 3: Backend API (FastAPI)
- [ ] **Environment Setup**: Define requirements for a GPU-enabled inference server.
- [ ] **Model Loading**: Implement logic to load the base Llama 3.2 model and the PEFT adapters.
- [ ] **Chat Endpoint**: Create the `/v1/chat/completions` endpoint with streaming support.
- [ ] **Adapter Switching**: Implement the dynamic persona selection logic.

### Phase 4: Frontend Development (React + TypeScript)
- [ ] **Vite Setup**: Bootstrap the React application.
- [ ] **Thematic UI**: 
    - [ ] Create the "Holocron" (Sith) theme with glowing red aesthetics.
    - [ ] Create the "Terminal" (Droid) theme with utilitarian yellow aesthetics.
- [ ] **Chat Logic**: Implement the streaming response handler and message history management.

### Phase 5: Final Validation
- [ ] **End-to-End Test**: Verify the full flow from UI to fine-tuned model.
- [ ] **Persona Tuning**: Adjust system prompts if necessary to sharpen the "cadence and philosophy" alignment.
