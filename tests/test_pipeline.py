"""Tests for corpus discovery and index building.

No real embeddings here -- build_index takes a VectorStoreManager, so a stub
stands in for it. The point of these tests is the decision-making around a
partial corpus, which is where this project has actually failed.
"""

import json

import pytest
from langchain_core.documents import Document

from src.ingestion.pipeline import (
    CorpusReport,
    build_index,
    chunk_id,
    collect_documents,
    plan_sync,
    sync_index,
)


class StubManager:
    """Records what it was asked to index without embedding anything."""

    def __init__(self, existing=None, loadable=None):
        self.added = None
        self.added_ids = None
        self.deleted = []
        self.saved = False
        self.save_count = 0
        self._existing = set(existing or ())
        # Default: an index exists iff we were given ids for it.
        self._loadable = bool(existing) if loadable is None else loadable

    def load(self):
        return self._loadable

    def known_ids(self):
        return set(self._existing)

    def add_documents(self, documents, ids=None):
        self.added = documents
        self.added_ids = ids

    def delete(self, ids):
        self.deleted = list(ids)

    def save(self):
        self.saved = True
        self.save_count += 1


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


def test_build_index_labels_documents_with_content_ids():
    """Without ids in the index, no later build could diff against it."""
    manager = StubManager()
    docs = [Document(page_content="a", metadata={"type": "lore"})]

    build_index(docs, vs_manager=manager)

    assert manager.added_ids == [chunk_id(docs[0])]


def test_build_index_does_not_hand_faiss_duplicate_ids():
    """The real corpus has 1963 documents and 1908 distinct ids.

    A repeated docstore key leaves FAISS with more vectors than documents, and
    the duplicate then reads as new on every incremental sync.
    """
    manager = StubManager()
    doc = Document(page_content="repeated line", metadata={"type": "lore"})

    build_index([doc, doc], vs_manager=manager)

    assert manager.added_ids == [chunk_id(doc)]
    assert len(manager.added) == 1


def test_build_index_refuses_an_empty_document_list():
    """An empty index loads cleanly and answers everything with nothing."""
    manager = StubManager()

    with pytest.raises(ValueError, match="zero documents"):
        build_index([], vs_manager=manager)

    assert not manager.saved


# --- chunk identity --------------------------------------------------------


def lore_doc(text, title="Rule of Two"):
    return Document(page_content=text, metadata={"type": "lore", "title": title})


def test_chunk_id_is_stable_for_identical_content():
    assert chunk_id(lore_doc("Darth Bane")) == chunk_id(lore_doc("Darth Bane"))


def test_chunk_id_changes_with_content():
    assert chunk_id(lore_doc("Darth Bane")) != chunk_id(lore_doc("Darth Plagueis"))


def test_chunk_id_distinguishes_same_text_from_different_speakers():
    vader = Document(page_content="Yes.", metadata={"type": "dialogue", "character": "VADER"})
    leia = Document(page_content="Yes.", metadata={"type": "dialogue", "character": "LEIA"})

    assert chunk_id(vader) != chunk_id(leia)


def test_chunk_id_ignores_source_path():
    """Source is machine-specific; including it would make every chunk look new."""
    here = Document(page_content="x", metadata={"type": "lore", "source": "C:/repo/a.json"})
    there = Document(page_content="x", metadata={"type": "lore", "source": "/ci/b.json"})

    assert chunk_id(here) == chunk_id(there)


def test_field_boundaries_cannot_be_shifted_to_forge_a_collision():
    """("ab","c") and ("a","bc") must not hash alike -- hence the separator."""
    a = Document(page_content="c", metadata={"type": "lore", "title": "ab"})
    b = Document(page_content="bc", metadata={"type": "lore", "title": "a"})

    assert chunk_id(a) != chunk_id(b)


# --- sync planning ---------------------------------------------------------


def test_plan_sync_on_a_fresh_index_adds_everything():
    docs = [lore_doc("a"), lore_doc("b")]
    plan = plan_sync(docs, known_ids=set())

    assert len(plan.added) == 2
    assert plan.removed == []
    assert plan.unchanged == 0
    assert plan.changed


def test_plan_sync_reembeds_nothing_when_the_corpus_is_unchanged():
    """The whole reason this exists: adding one page must not re-embed the rest."""
    docs = [lore_doc("a"), lore_doc("b")]
    plan = plan_sync(docs, known_ids={chunk_id(d) for d in docs})

    assert plan.added == []
    assert plan.removed == []
    assert plan.unchanged == 2
    assert not plan.changed


def test_plan_sync_adds_only_the_new_chunk():
    old = [lore_doc("a"), lore_doc("b")]
    new = [*old, lore_doc("c")]
    plan = plan_sync(new, known_ids={chunk_id(d) for d in old})

    assert [d.page_content for d in plan.added] == ["c"]
    assert plan.unchanged == 2


def test_plan_sync_removes_chunks_that_left_the_corpus():
    gone = lore_doc("deleted passage")
    kept = lore_doc("kept passage")
    plan = plan_sync([kept], known_ids={chunk_id(gone), chunk_id(kept)})

    assert plan.removed == [chunk_id(gone)]
    assert plan.added == []
    assert plan.unchanged == 1


def test_an_edited_passage_is_one_add_and_one_remove():
    before = lore_doc("Bane decreed it around 1032 BBY")
    after = lore_doc("Bane decreed it around 1000 BBY")
    plan = plan_sync([after], known_ids={chunk_id(before)})

    assert len(plan.added) == 1
    assert plan.removed == [chunk_id(before)]
    assert plan.unchanged == 0


def test_duplicate_chunks_are_collapsed_to_one_id():
    """Chunk overlap and repeated screenplay lines both produce duplicates.

    FAISS ids must be unique, and a duplicate left in the add list would be
    re-added on every single sync -- a rebuild that never converges.
    """
    plan = plan_sync([lore_doc("same"), lore_doc("same")], known_ids=set())

    assert len(plan.added) == 1


def test_plan_describe_reports_all_three_counts():
    plan = plan_sync([lore_doc("a"), lore_doc("b")], known_ids={chunk_id(lore_doc("a"))})
    assert plan.describe() == "1 new, 0 removed, 1 unchanged"


# --- sync execution --------------------------------------------------------


def test_sync_falls_back_to_a_full_build_when_no_index_exists():
    manager = StubManager(loadable=False)
    docs = [lore_doc("a")]

    _, plan = sync_index(docs, vs_manager=manager)

    assert plan.added == docs
    assert manager.saved


def test_sync_adds_and_deletes_against_an_existing_index():
    gone = lore_doc("old")
    kept = lore_doc("kept")
    manager = StubManager(existing={chunk_id(gone), chunk_id(kept)})

    _, plan = sync_index([kept, lore_doc("new")], vs_manager=manager)

    assert [d.page_content for d in manager.added] == ["new"]
    assert manager.added_ids == [chunk_id(lore_doc("new"))]
    assert manager.deleted == [chunk_id(gone)]
    assert plan.unchanged == 1


def test_sync_does_not_rewrite_the_index_when_nothing_changed():
    """save() re-digests the index for the manifest; a no-op save churns it."""
    docs = [lore_doc("a")]
    manager = StubManager(existing={chunk_id(docs[0])})

    _, plan = sync_index(docs, vs_manager=manager)

    assert not plan.changed
    assert manager.save_count == 0
    assert manager.deleted == []


def test_prune_false_keeps_chunks_the_local_corpus_is_missing():
    """A rate-limited scrape or unfetched corpora must not delete real data."""
    kept = lore_doc("kept")
    absent = lore_doc("page the scrape failed to return")
    manager = StubManager(existing={chunk_id(kept), chunk_id(absent)})

    _, plan = sync_index([kept], vs_manager=manager, prune=False)

    assert plan.removed == []
    assert manager.deleted == []
    assert not plan.changed
    # "nothing to prune" and "pruning was suppressed" must stay distinguishable;
    # a partial local corpus can sit in the second state indefinitely.
    assert plan.retained == [chunk_id(absent)]
    assert "1 kept (pruning off)" in plan.describe()


def test_describe_stays_quiet_about_retention_when_nothing_was_retained():
    plan = plan_sync([lore_doc("a")], known_ids=set())
    assert "kept" not in plan.describe()


def test_sync_refuses_an_empty_corpus():
    manager = StubManager(existing={"whatever"})

    with pytest.raises(ValueError, match="zero documents"):
        sync_index([], vs_manager=manager)

    assert manager.deleted == []
    assert not manager.saved
