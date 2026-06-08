# Engineering & Validation Protocol (Sith Holocron)

## 1. Development Lifecycle
Every feature or bug fix must follow the **Red-Green-Refactor** pattern:
1.  **Reproduction/Test First**: Create a test case (unit or integration) that defines the expected behavior.
2.  **Implementation**: Write the minimal code required to satisfy the test.
3.  **Verification**: Run the full test suite and linting.
4.  **Documentation**: Update relevant docs/comments.

## 2. Testing Standards
- **Unit Tests**: Mandatory for all utility functions, parsers, and logic-heavy components. (Use `pytest`).
- **Integration Tests**: Required for vector database interactions and LLM prompt construction.
- **E2E Validation**: The application must be run end-to-end (from data ingestion to query response) before any push to `master`.
- **Edge Case Coverage**:
    - Empty/malformed data files.
    - Retrieval failures (no matches found).
    - Character persona "leakage" (ensure Sith voice remains consistent).
    - Unicode/Special character handling in Star Wars names (e.g., Thrawn's full name).

## 3. Commit & Push Gatekeeping
Before every `git commit`, the following must be true:
- [ ] `pytest` passes with 0 failures.
- [ ] No hardcoded secrets or API keys.
- [ ] The specific change is verified via a script or manual CLI check.
- [ ] The `PROTOCOL.md` and `WORKPLAN.md` are updated if necessary.

## 4. LLM & RAG Quality Control
- **Context Injection**: Log the retrieved snippets to ensure relevant data is being passed to the model.
- **Persona Audit**: Use a "Persona Consistency" check—does the output sound like the target Sith Lord?
- **Grounding**: Ensure the AI cites or uses specific facts from the Wookieepedia lore rather than hallucinating.
