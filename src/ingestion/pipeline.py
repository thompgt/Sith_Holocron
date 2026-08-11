"""Corpus discovery and index building, with no CLI and no LLM attached.

This logic lived inside SithHolocronCLI._bootstrap_data, which meant building an
index required a TTY and a GOOGLE_API_KEY -- the CLI constructs the Gemini
wrapper and exits before it ever reaches indexing if the key is absent. Indexing
touches neither. Anything that needs an index but not a conversation (the
retrieval eval, a CI job, a rebuild after a re-scrape) had no way in.

So the corpus paths, the "which corpora are missing" reporting, and the build
itself live here. src/main.py keeps the console output; this module returns
findings instead of printing them.
"""

import hashlib
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document

from src.ingestion.lore_processor import LoreProcessor
from src.ingestion.script_parser import ScriptParser
from src.retrieval.vector_store import VectorStoreManager

#: Scraped by scripts/scrape_wookieepedia.py. Tracked in the repo, so normally
#: present -- but it holds whatever the last successful scrape produced, and
#: Fandom rate-limits, so it is routinely a subset of the pages requested.
LORE_PATH = "data/raw/lore.json"

#: Third-party screenplay corpora, fetched by scripts/fetch_corpora.py. Absent
#: these the index is lore-only and every persona's dialogue half is empty.
OT_SCRIPT_PATH = "data/raw/star-wars-scripts/Text_files/EpisodeIV_dialogues.txt"
PREQUEL_CSV_PATH = "data/raw/prequel-csv/star_wars_1_data.csv"


@dataclass
class CorpusReport:
    """What was found, what was not, and what that means for the index."""

    documents: list[Document] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def dialogue_count(self) -> int:
        return sum(1 for doc in self.documents if doc.metadata.get("type") == "dialogue")

    @property
    def lore_count(self) -> int:
        return len(self.documents) - self.dialogue_count

    @property
    def is_empty(self) -> bool:
        return not self.documents

    @property
    def is_lore_only(self) -> bool:
        """True when an index built from this would silently have no voices."""
        return bool(self.documents) and self.dialogue_count == 0


def collect_documents(
    lore_path: str = LORE_PATH,
    ot_script_path: str = OT_SCRIPT_PATH,
    prequel_csv_path: str = PREQUEL_CSV_PATH,
) -> CorpusReport:
    """Read every corpus that is present. Absent ones are reported, not raised.

    A missing corpus is a normal state here -- the screenplay corpora are
    submodules a fresh clone has not fetched -- so the caller decides whether to
    proceed. What must never happen is proceeding *quietly*, which is why the
    report distinguishes empty from lore-only.
    """
    report = CorpusReport()
    lore_processor = LoreProcessor()
    script_parser = ScriptParser()

    if os.path.exists(lore_path):
        report.documents.extend(lore_processor.process_file(lore_path))
    else:
        report.missing.append(lore_path)

    if os.path.exists(ot_script_path):
        report.documents.extend(script_parser.parse_tab_txt(ot_script_path))
    else:
        report.missing.append(ot_script_path)

    if os.path.exists(prequel_csv_path):
        report.documents.extend(
            script_parser.parse_csv(
                prequel_csv_path, sep=";", char_col="from", text_col="text"
            )
        )
    else:
        report.missing.append(prequel_csv_path)

    return report


def build_index(
    documents: list[Document],
    persist_directory: str = "data/vector_store",
    vs_manager: VectorStoreManager | None = None,
) -> VectorStoreManager:
    """Embed documents into a fresh index and persist it with its manifest.

    Refuses an empty document list rather than writing an empty index: an empty
    index loads without error and answers every query with nothing, which is the
    single hardest failure in this system to notice from the outside.
    """
    if not documents:
        raise ValueError(
            "Refusing to build an index from zero documents -- it would load "
            "cleanly and silently answer every query with nothing."
        )

    manager = vs_manager or VectorStoreManager(persist_directory=persist_directory)
    unique = dedupe(documents)
    manager.add_documents(unique, ids=[chunk_id(doc) for doc in unique])
    manager.save()
    return manager


# --- incremental rebuilds --------------------------------------------------

#: Metadata fields that participate in a chunk's identity. Deliberately not the
#: whole metadata dict: "source" is an absolute-ish path that differs between
#: machines and CI, and including it would make every chunk look new on any
#: checkout at a different path.
IDENTITY_FIELDS = ("type", "title", "character")


def chunk_id(document: Document) -> str:
    """A stable id derived from a chunk's content and identifying metadata.

    Content-derived rather than positional: chunk #7 of an article is a
    different chunk after the article is re-scraped and re-split, but the *text*
    of an unchanged passage hashes the same either way. That is what lets a
    rebuild re-embed only what actually changed.

    The trailing separator matters -- without one, ("ab", "c") and ("a", "bc")
    hash identically and two distinct chunks would collide into one id.
    """
    digest = hashlib.sha256()
    for field_name in IDENTITY_FIELDS:
        digest.update(str(document.metadata.get(field_name) or "").encode("utf-8"))
        digest.update(b"\x00")
    digest.update(document.page_content.encode("utf-8"))
    return digest.hexdigest()


def dedupe(documents: list[Document]) -> list[Document]:
    """One document per chunk id, first occurrence kept.

    The corpora really do contain duplicates: chunk overlap can emit an
    identical passage twice and a screenplay repeats short lines verbatim. The
    real corpus in this repo yields 1,963 documents and 1,908 distinct ids.

    Duplicates must not reach FAISS. Its docstore is keyed by id, so a repeated
    id leaves the vector count higher than the docstore count -- and in an
    incremental sync the duplicate would be counted as new and re-added on every
    run, a rebuild that never converges.
    """
    seen: dict[str, Document] = {}
    for document in documents:
        seen.setdefault(chunk_id(document), document)
    return list(seen.values())


@dataclass
class SyncPlan:
    """What an incremental rebuild would change, before it changes it."""

    added: list[Document] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    #: Ids the corpus no longer contains that prune=False chose to keep. Held
    #: separately rather than by clearing `removed`, so "nothing to prune" and
    #: "pruning was suppressed" stay distinguishable -- the second is a state a
    #: partial local corpus can sit in for a long time without noticing.
    retained: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)

    @property
    def is_total_churn(self) -> bool:
        """Everything in the index is being replaced -- almost never intended.

        The realistic cause is not a changed corpus but an index built before
        chunk ids were content-derived: it holds UUID keys, so no id can match
        and a "sync" re-embeds the whole corpus while deleting the old one. That
        is correct but slower than a plain rebuild, so it is worth saying.
        """
        return bool(self.removed) and self.unchanged == 0

    def describe(self) -> str:
        text = (
            f"{len(self.added)} new, {len(self.removed)} removed, "
            f"{self.unchanged} unchanged"
        )
        if self.retained:
            text += f", {len(self.retained)} kept (pruning off)"
        return text


def plan_sync(documents: list[Document], known_ids: set[str]) -> SyncPlan:
    """Diff a freshly-read corpus against the ids already in an index.

    Pure: takes the id set, returns the plan, touches nothing. Kept separate
    from apply_sync so a caller can show the plan before spending money and
    minutes on embeddings -- and so the diff logic is testable without FAISS.
    """
    wanted = {chunk_id(document): document for document in dedupe(documents)}
    new_ids = wanted.keys() - known_ids

    return SyncPlan(
        added=[document for cid, document in wanted.items() if cid in new_ids],
        removed=sorted(known_ids - wanted.keys()),
        unchanged=len(wanted.keys() & known_ids),
    )


def sync_index(
    documents: list[Document],
    persist_directory: str = "data/vector_store",
    vs_manager: VectorStoreManager | None = None,
    prune: bool = True,
) -> tuple[VectorStoreManager, SyncPlan]:
    """Bring an existing index in line with the corpus, re-embedding only deltas.

    Falls back to a full build when there is no index yet. With prune=False,
    chunks that have left the corpus stay in the index -- useful when the corpus
    on this machine is knowingly partial (a rate-limited scrape, or corpora that
    were not fetched) and dropping what is missing would lose real data.
    """
    if not documents:
        raise ValueError(
            "Refusing to sync against zero documents -- with prune=True that "
            "would empty the index, and an empty index answers every query "
            "with nothing without erroring."
        )

    manager = vs_manager or VectorStoreManager(persist_directory=persist_directory)
    if not manager.load():
        # Report the plan build_index will actually carry out, deduped -- not the
        # raw corpus, or the caller's "N new" would not match the index's size.
        return build_index(documents, vs_manager=manager), SyncPlan(added=dedupe(documents))

    plan = plan_sync(documents, manager.known_ids())
    if not prune:
        plan.retained, plan.removed = plan.removed, []

    if plan.added:
        manager.add_documents(plan.added, ids=[chunk_id(doc) for doc in plan.added])
    manager.delete(plan.removed)

    # Only rewrite when something moved: save() re-digests the index files for
    # the manifest, and a no-op save would churn them for nothing.
    if plan.changed:
        manager.save()

    return manager, plan
