import pytest
import os
import shutil
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStoreManager

@pytest.fixture
def vector_manager(tmp_path):
    db_path = tmp_path / "test_faiss"
    manager = VectorStoreManager(persist_directory=str(db_path))
    return manager

def test_add_documents_and_search(vector_manager):
    docs = [
        Document(page_content="The Rule of Two mandates only two Sith.", metadata={"title": "Rule of Two"}),
        Document(page_content="Peace is a lie, there is only passion.", metadata={"title": "Sith Code"})
    ]
    
    vector_manager.add_documents(docs)
    results = vector_manager.search("How many Sith should there be?", k=1)
    
    assert len(results) == 1
    assert "Rule of Two" in results[0].metadata["title"]

def test_persistence(vector_manager, tmp_path):
    db_path = str(tmp_path / "persist_faiss")
    manager1 = VectorStoreManager(persist_directory=db_path)
    docs = [Document(page_content="Luke, I am your father.", metadata={"char": "Vader"})]
    
    manager1.add_documents(docs)
    manager1.save()
    
    # Reload in a new instance
    manager2 = VectorStoreManager(persist_directory=db_path)
    manager2.load()
    results = manager2.search("Who is the father?", k=1)
    
    assert len(results) == 1
    assert results[0].metadata["char"] == "Vader"
