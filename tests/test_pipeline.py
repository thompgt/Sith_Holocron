"""Tests for corpus discovery and index building.

No real embeddings here -- build_index takes a VectorStoreManager, so a stub
stands in for it. The point of these tests is the decision-making around a
partial corpus, which is where this project has actually failed.
"""

import json

import pytest
from langchain_core.documents import Document

from src.ingestion.pipeline import CorpusReport, build_index, collect_documents


class StubManager:
    """Records what it was asked to index without embedding anything."""

    def __init__(self):
        self.added = None
        self.saved = False

    def add_documents(self, documents):
        self.added = documents

    def save(self):
        self.saved = True


@pytest.fixture
def corpora(tmp_path):
    """A full set of corpora: lore, a tab-separated script, a prequel CSV."""
    lore = tmp_path / "lore.json"
    lore.write_text(
        json.dumps(
            [{"url": "u", "title": "Rule of Two", "content": "Darth Bane decreed it."}]
        ),
        encoding="utf-8",
    )

    script = tmp_path / "EpisodeIV_dialogues.txt"
    script.write_text(
        "VADER\tI find your lack of faith disturbing.\n"
        "LEIA\tDarth Vader. Only you could be so bold.\n",
        encoding="utf-8",
    )

    csv = tmp_path / "prequel.csv"
    csv.write_text("from;text\nDARTH SIDIOUS;Everything is proceeding as I have foreseen.\n",
                   encoding="utf-8")

    return {
        "lore_path": str(lore),
        "ot_script_path": str(script),
        "prequel_csv_path": str(csv),
    }


# --- discovery -------------------------------------------------------------


def test_collects_every_corpus_that_is_present(corpora):
    report = collect_documents(**corpora)

    assert not report.missing
    assert report.lore_count >= 1
    assert report.dialogue_count == 3
    assert not report.is_lore_only


def test_missing_corpora_are_reported_not_raised(corpora):
    report = collect_documents(
        lore_path=corpora["lore_path"],
        ot_script_path="does/not/exist.txt",
        prequel_csv_path="also/absent.csv",
    )

    # A fresh clone has not fetched the submodules; that is a normal state and
    # must not be an exception, or the CLI could never offer to fix it.
    assert report.missing == ["does/not/exist.txt", "also/absent.csv"]
    assert report.documents


def test_a_clone_with_no_corpora_at_all_is_empty_not_lore_only(corpora):
    report = collect_documents(
        lore_path="nope.json", ot_script_path="nope.txt", prequel_csv_path="nope.csv"
    )

    assert report.is_empty
    assert len(report.missing) == 3
    # is_lore_only must be False here: "no data" and "data but no voices" call
    # for different messages, and collapsing them loses the useful one.
    assert not report.is_lore_only


def test_lore_without_scripts_is_flagged_as_lore_only(corpora):
    report = collect_documents(
        lore_path=corpora["lore_path"],
        ot_script_path="absent.txt",
        prequel_csv_path="absent.csv",
    )

    assert report.is_lore_only
    assert report.dialogue_count == 0
    assert report.lore_count > 0


def test_counts_partition_the_documents():
    report = CorpusReport(
        documents=[
            Document(page_content="a", metadata={"type": "lore"}),
            Document(page_content="b", metadata={"type": "dialogue", "character": "VADER"}),
            # Untyped, as chunks from an index built before LoreProcessor
            # started stamping type. Must count as lore, not vanish.
            Document(page_content="c", metadata={}),
        ]
    )

    assert report.dialogue_count == 1
    assert report.lore_count == 2
    assert report.lore_count + report.dialogue_count == len(report.documents)


# --- building --------------------------------------------------------------


def test_build_index_adds_then_saves():
    manager = StubManager()
    docs = [Document(page_content="a", metadata={"type": "lore"})]

    assert build_index(docs, vs_manager=manager) is manager
    assert manager.added == docs
    assert manager.saved


def test_build_index_refuses_an_empty_document_list():
    """An empty index loads cleanly and answers everything with nothing."""
    manager = StubManager()

    with pytest.raises(ValueError, match="zero documents"):
        build_index([], vs_manager=manager)

    assert not manager.saved
