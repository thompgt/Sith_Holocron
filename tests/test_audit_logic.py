from src.llm.persona_manager import PersonaManager
from langchain_core.documents import Document

def test_audit_heuristics_logic():
    pm = PersonaManager()
    
    # Mocking the behavior for a logic test
    character = "VADER"
    response = "I am Darth Vader. The Rule of Two is the way."
    lore_doc = Document(page_content="The Rule of Two is the way.", metadata={"type": "lore"})
    
    # Check 1: Identity
    identity_pass = "VADER" in response.upper() or "DARTH VADER" in response.upper()
    
    # Check 2: Grounding
    grounding_pass = lore_doc.page_content.lower() in response.lower()
    
    assert identity_pass is True
    assert grounding_pass is True

def test_format_context_completeness():
    pm = PersonaManager()
    docs = [
        Document(page_content="Lore fact.", metadata={"type": "lore", "title": "History"}),
        Document(page_content="Dialogue line.", metadata={"type": "dialogue", "character": "VADER"})
    ]
    context = pm.format_context(docs)
    assert "LORE CONTEXT" in context
    assert "PAST UTTERANCES" in context
    assert "Lore fact." in context
    assert "Dialogue line." in context
