"""Build the FAISS index from whatever corpora are on disk.

    python scripts/build_index.py
    python scripts/build_index.py --require-dialogue      # fail if no corpora
    python scripts/build_index.py --index-dir /tmp/scratch-index

Previously the only way to build an index was to start the interactive CLI,
which constructs the Gemini wrapper first and exits when GOOGLE_API_KEY is
absent -- so a machine with corpora but no API key could not index them, and
neither could CI. Indexing needs no key and no TTY.

Exit codes:
  0  index built
  1  nothing to index, or a required corpus was missing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import build_index, collect_documents  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", default="data/vector_store")
    parser.add_argument(
        "--require-dialogue",
        action="store_true",
        help=(
            "exit 1 rather than building a lore-only index. A lore-only index "
            "loads fine and answers every query with no dialogue, so use this "
            "anywhere the result is consumed unattended."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = collect_documents()

    for path in report.missing:
        print(f"missing corpus: {path}", file=sys.stderr)
    if report.missing:
        print(
            "Run `python scripts/fetch_corpora.py` to fetch the screenplay corpora.",
            file=sys.stderr,
        )

    if report.is_empty:
        print("No corpora found at all; nothing to index.", file=sys.stderr)
        return 1

    if report.is_lore_only:
        message = (
            "No dialogue documents -- every persona's voice half will be empty."
        )
        if args.require_dialogue:
            print(f"{message} --require-dialogue was set, so refusing.", file=sys.stderr)
            return 1
        print(f"WARNING: {message}", file=sys.stderr)

    print(
        f"Indexing {len(report.documents)} documents "
        f"({report.lore_count} lore, {report.dialogue_count} dialogue)..."
    )
    manager = build_index(report.documents, persist_directory=args.index_dir)
    print(f"Wrote {manager.index_path} and {manager.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
