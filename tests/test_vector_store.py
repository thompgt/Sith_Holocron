import pytest
import os
import shutil
from langchain_core.documents import Document
from src.retrieval.vector_store import (
    TRUST_ENV_VAR,
    IndexTrustError,
    VectorStoreManager,
)


@pytest.fixture(autouse=True)
def _no_trust_override(monkeypatch):
    """Keep an operator's real HOLOCRON_TRUST_INDEX out of the test run."""
    monkeypatch.delenv(TRUST_ENV_VAR, raising=False)


@pytest.fixture
def saved_index(tmp_path):
    """A manager with an index saved to disk, plus its manifest."""
    path = str(tmp_path / "trust_faiss")
    manager = VectorStoreManager(persist_directory=path)
    manager.add_documents([Document(page_content="The dark side of the Force.")])
    manager.save()
    return path

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


def test_save_writes_a_manifest(saved_index):
    manager = VectorStoreManager(persist_directory=saved_index)
    assert os.path.exists(manager.manifest_path)


def test_load_rejects_a_tampered_docstore(saved_index):
    # The .pkl is the unpickled-at-load file, so this is the file an attacker
    # swaps to get code execution at server import.
    manager = VectorStoreManager(persist_directory=saved_index)
    with open(manager.docstore_path, "ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(IndexTrustError, match="do not match the manifest"):
        manager.load()

    assert manager.vector_store is None


def test_load_rejects_an_index_with_no_manifest(saved_index):
    manager = VectorStoreManager(persist_directory=saved_index)
    os.remove(manager.manifest_path)

    with pytest.raises(IndexTrustError, match="No manifest"):
        manager.load()


def test_load_rejects_an_unreadable_manifest(saved_index):
    manager = VectorStoreManager(persist_directory=saved_index)
    with open(manager.manifest_path, "w", encoding="utf-8") as handle:
        handle.write("{not json")

    with pytest.raises(IndexTrustError, match="unreadable"):
        manager.load()


def test_trust_override_allows_a_tampered_index(saved_index, monkeypatch):
    manager = VectorStoreManager(persist_directory=saved_index)
    os.remove(manager.manifest_path)
    monkeypatch.setenv(TRUST_ENV_VAR, "1")

    assert manager.load() is True


def test_missing_index_returns_false_rather_than_raising(tmp_path):
    # A absent index and an unverifiable one are different situations; only the
    # latter is an error.
    manager = VectorStoreManager(persist_directory=str(tmp_path / "nothing_here"))
    assert manager.load() is False


def test_verified_index_loads(saved_index):
    manager = VectorStoreManager(persist_directory=saved_index)
    assert manager.load() is True
    assert len(manager.search("dark side", k=1)) == 1
