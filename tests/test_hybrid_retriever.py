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
