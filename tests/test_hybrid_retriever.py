import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever


class MockVectorStore:
    def search(self, query, k=4):
        # Return a mix of lore and dialogue
        return [
            Document(page_content="Fact about Sith.", metadata={"type": "lore", "title": "History"}),
            Document(page_content="Dialogue from Vader.", metadata={"type": "dialogue", "character": "VADER"})
        ]

@pytest.fixture
def hybrid_retriever():
    mock_vs = MockVectorStore()
    return HybridRetriever(vector_store_manager=mock_vs)

def test_hybrid_retrieval_balance(hybrid_retriever):
    # Test that it fetches results
    results = hybrid_retriever.retrieve("Sith history", k=2)
    assert len(results) == 2

    # Check if we have both types (or at least the structure is respected)
    types = [doc.metadata.get("type") for doc in results]
    assert "lore" in types or "dialogue" in types

def test_persona_filtering_logic(hybrid_retriever):
    # If we filter by VADER, we should prioritize or only see VADER dialogue
    results = hybrid_retriever.retrieve("plans", character="VADER")
    for doc in results:
        if doc.metadata.get("type") == "dialogue":
            assert doc.metadata.get("character") == "VADER"


class AliasVectorStore:
    """Dialogue credited the way the corpora actually credit it."""

    def __init__(self, speakers):
        self.speakers = speakers

    def search(self, query, k=4):
        return [
            Document(
                page_content=f"Line by {s}.",
                metadata={"type": "dialogue", "character": s},
            )
            for s in self.speakers
        ]


def _speakers(docs):
    return {d.metadata["character"] for d in docs if d.metadata.get("type") == "dialogue"}


def test_palpatine_matches_his_script_credits():
    # The prequels credit him DARTH SIDIOUS and the OT credits him EMPEROR;
    # nothing in either corpus is attributed to the persona key "PALPATINE"
    # alone. Matching the key by equality returned zero dialogue documents.
    vs = AliasVectorStore(["DARTH SIDIOUS", "EMPEROR", "LUKE"])
    retriever = HybridRetriever(vector_store_manager=vs)

    docs = retriever.retrieve("power", character="PALPATINE", k=4)

    assert _speakers(docs) == {"DARTH SIDIOUS", "EMPEROR"}


def test_generic_sith_matches_any_dark_side_speaker():
    # No line is ever spoken by a character named GENERIC_SITH, so this persona
    # could never match anything at all.
    vs = AliasVectorStore(["DARTH MAUL", "VADER", "OBI-WAN"])
    retriever = HybridRetriever(vector_store_manager=vs)

    docs = retriever.retrieve("the dark side", character="GENERIC_SITH", k=4)

    assert _speakers(docs) == {"DARTH MAUL", "VADER"}


def test_unknown_persona_falls_back_instead_of_matching_nothing():
    vs = AliasVectorStore(["VADER", "JAR JAR"])
    retriever = HybridRetriever(vector_store_manager=vs)

    docs = retriever.retrieve("anything", character="NOT_A_PERSONA", k=4)

    assert _speakers(docs) == {"VADER"}


class SplitVectorStore:
    """One lore surplus and one dialogue surplus, neither enough on its own."""

    def search(self, query, k=4):
        return [
            Document(page_content="lore 1", metadata={"type": "lore"}),
            Document(page_content="lore 2", metadata={"type": "lore"}),
            Document(page_content="lore 3", metadata={"type": "lore"}),
            Document(
                page_content="line 1", metadata={"type": "dialogue", "character": "VADER"}
            ),
            Document(
                page_content="line 2", metadata={"type": "dialogue", "character": "VADER"}
            ),
            Document(
                page_content="line 3", metadata={"type": "dialogue", "character": "VADER"}
            ),
        ]


@pytest.mark.parametrize("lore_weight", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_backfill_returns_k_whenever_the_pools_can_cover_it(k, lore_weight):
    """Backfill must saturate k regardless of how the targets split.

    The store holds 3 lore and 3 dialogue, so every k up to 6 is coverable no
    matter which way lore_weight skews the targets. This pins the property the
    backfill exists to provide -- a short pool gets topped up from the other one.
    """
    retriever = HybridRetriever(vector_store_manager=SplitVectorStore())

    docs = retriever.retrieve("q", character="VADER", k=k, lore_weight=lore_weight)

    assert len(docs) == k
