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
    manager.add_documents(documents)
    manager.save()
    return manager
