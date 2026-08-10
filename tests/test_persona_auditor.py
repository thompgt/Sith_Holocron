"""Unit tests for PersonaAuditor.

The auditor previously lived at tests/persona_audit.py with no test_ functions
in it, so pytest collected nothing from the file and it went unexercised. These
drive it entirely through stubs -- no index, no Gemini call, no network.
"""

import pytest
from langchain_core.documents import Document

from src.eval.persona_auditor import PersonaAuditor, summarize
from src.llm.persona_manager import PersonaManager


class StubRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def retrieve(self, query, character=None, k=4):
        self.calls.append((query, character, k))
        return self.docs


class StubLLM:
    def __init__(self, response):
        self.response = response

    def chat(self, system_prompt, query, context):
        return self.response


def lore(text):
    return Document(page_content=text, metadata={"type": "lore"})


def dialogue(text, character="VADER"):
    return Document(
        page_content=text, metadata={"type": "dialogue", "character": character}
    )


def make_auditor(response, docs=()):
    return PersonaAuditor(StubLLM(response), StubRetriever(list(docs)), PersonaManager())


# --- the bug that was actually in the file --------------------------------


def test_unknown_character_does_not_raise():
    """The old code did `self.pm.personas.get(name)["name"]`.

    An unknown character makes .get() return None and subscripting it raises
    TypeError, so auditing anything outside the persona table crashed instead of
    reporting a failed check.
    """
    auditor = make_auditor("Some response about the Force and nothing else.")
    result = auditor.audit_response("JAR_JAR", "who are you?")
    assert result["character_check"] is False


def test_unknown_character_still_matches_generic_sith_aliases():
    """aliases() falls back to GENERIC_SITH, so a dark-side name still counts."""
    auditor = make_auditor("Darth Maul says nothing of consequence here.")
    assert auditor.audit_response("NOBODY", "q")["character_check"] is True


def test_empty_character_does_not_raise():
    auditor = make_auditor("A response.")
    assert auditor.audit_response("", "q")["character_check"] is False


# --- character check ------------------------------------------------------


def test_character_name_in_response_matches():
    auditor = make_auditor("VADER will not be denied.")
    assert auditor.audit_response("VADER", "q")["character_check"] is True


def test_character_check_is_case_insensitive():
    auditor = make_auditor("vader will not be denied.")
    assert auditor.audit_response("VADER", "q")["character_check"] is True


def test_alias_counts_as_in_character():
    """The old two-way comparison missed aliases entirely."""
    auditor = make_auditor("Sidious foresaw this outcome.")
    assert auditor.audit_response("PALPATINE", "q")["character_check"] is True


def test_unrelated_response_is_not_in_character():
    auditor = make_auditor("The weather on Naboo is pleasant today.")
    assert auditor.audit_response("VADER", "q")["character_check"] is False


# --- grounding check ------------------------------------------------------


def test_context_used_when_lore_prefix_appears():
    passage = "The Rule of Two governs the Sith order completely."
    auditor = make_auditor(
        f"Indeed. {passage} That is the way.", docs=[lore(passage)]
    )
    assert auditor.audit_response("VADER", "q")["context_used"] is True


def test_dialogue_documents_do_not_count_as_grounding():
    """Only lore is checked; dialogue is voice, not evidence."""
    line = "I find your lack of faith disturbing right now."
    auditor = make_auditor(f"As I said: {line}", docs=[dialogue(line)])
    assert auditor.audit_response("VADER", "q")["context_used"] is False


def test_empty_lore_document_does_not_match_everything():
    """`"" in response` is always True, so an empty passage would score as
    grounded against any response at all."""
    auditor = make_auditor("Anything.", docs=[lore("")])
    assert auditor.audit_response("VADER", "q")["context_used"] is False


def test_no_documents_means_not_grounded():
    auditor = make_auditor("A confident answer with no sources.")
    assert auditor.audit_response("VADER", "q")["context_used"] is False


# --- plumbing -------------------------------------------------------------


def test_k_is_forwarded_to_the_retriever():
    retriever = StubRetriever([])
    auditor = PersonaAuditor(StubLLM("x"), retriever, PersonaManager())
    auditor.audit_response("VADER", "why?", k=7)
    assert retriever.calls == [("why?", "VADER", 7)]


@pytest.mark.parametrize(
    "response, expected",
    [("Yes.", False), ("A" * PersonaAuditor.MIN_SUBSTANTIVE_CHARS, True)],
)
def test_substantive_flags_non_answers(response, expected):
    auditor = make_auditor(response)
    assert auditor.audit_response("VADER", "q")["substantive"] is expected


def test_audit_all_and_summarize():
    auditor = make_auditor("VADER speaks, and the galaxy listens to every word.")
    results = auditor.audit_all({"VADER": ["a", "b"], "PALPATINE": ["c"]})
    assert [len(v) for v in results.values()] == [2, 1]

    summary = summarize(results)
    assert summary["total"] == 3
    # The stub always answers with the same VADER line, so the two VADER queries
    # score in-character and the PALPATINE one does not -- the summary is over
    # every query, not per persona.
    assert summary["in_character"] == pytest.approx(2 / 3)
    assert summary["grounded"] == 0.0


def test_summarize_handles_no_results():
    assert summarize({}) == {"total": 0}
