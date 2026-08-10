import pytest
from langchain_core.documents import Document

from src.llm.persona_manager import PersonaManager


@pytest.fixture
def persona_manager():
    return PersonaManager()

def test_get_persona_prompt_basic(persona_manager):
    prompt = persona_manager.get_system_prompt("VADER")
    assert "Darth Vader" in prompt
    assert "Sith" in prompt

def test_format_context_injection(persona_manager):
    docs = [
        Document(page_content="The Rule of Two is key.", metadata={"type": "lore", "title": "Rule of Two"}),
        Document(page_content="I find your lack of faith disturbing.", metadata={"type": "dialogue", "character": "VADER"})
    ]
    formatted = persona_manager.format_context(docs)

    assert "LORE CONTEXT" in formatted
    assert "The Rule of Two" in formatted
    assert "PAST UTTERANCES" in formatted
    assert "lack of faith" in formatted

def test_invalid_persona_fallback(persona_manager):
    # Should fallback to a generic Sith Lord if character unknown
    prompt = persona_manager.get_system_prompt("UNKNOWN_SITH")
    assert "Sith Lord" in prompt
