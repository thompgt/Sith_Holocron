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

from src.ingestion.pipeline import build_index, collect_documents, sync_index  # noqa: E402
from src.retrieval.vector_store import IndexTrustError  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", default="data/vector_store")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "diff the corpus against the existing index and embed only what "
            "changed, instead of rebuilding from scratch. Falls back to a full "
            "build when there is no index yet."
        ),
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help=(
            "with --incremental, keep chunks that are no longer in the local "
            "corpus. Use it when this machine's corpus is knowingly partial -- a "
            "rate-limited scrape, or corpora that were never fetched -- so a "
            "sync does not delete data it simply cannot see."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --incremental, report the diff and exit without embedding anything",
    )
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
        f"Read {len(report.documents)} documents "
        f"({report.lore_count} lore, {report.dialogue_count} dialogue)."
    )

    if not args.incremental:
        print("Building a fresh index...")
        manager = build_index(report.documents, persist_directory=args.index_dir)
        print(f"Wrote {manager.index_path} and {manager.manifest_path}")
        return 0

    try:
        if args.dry_run:
            return _report_dry_run(args, report)
        manager, plan = sync_index(
            report.documents,
            persist_directory=args.index_dir,
            prune=not args.no_prune,
        )
    except IndexTrustError as exc:
        # An incremental sync loads the existing index, so it is subject to the
        # same trust check. Do not offer to bypass it here.
        print(f"Existing index failed verification:\n{exc}", file=sys.stderr)
        return 1

    print(f"Sync: {plan.describe()}")
    _warn_on_total_churn(plan)
    if plan.changed:
        print(f"Wrote {manager.index_path} and {manager.manifest_path}")
    else:
        print("Index already matches the corpus; nothing rewritten.")
    return 0


def _warn_on_total_churn(plan) -> None:
    if plan.is_total_churn:
        print(
            "Every chunk in the index is being replaced. If the corpus did not "
            "actually change, the index predates content-addressed chunk ids "
            "and holds UUID keys that can never match -- drop --incremental "
            "once to rebuild it, after which syncs will be cheap.",
            file=sys.stderr,
        )


def _report_dry_run(args, report) -> int:
    """Show what a sync would change without embedding anything."""
    # Imported here rather than at module scope: the plan is pure and needs no
    # index, but constructing a VectorStoreManager loads the embedding model,
    # which is the cost a dry run exists to avoid paying twice.
    from src.ingestion.pipeline import plan_sync
    from src.retrieval.vector_store import VectorStoreManager

    manager = VectorStoreManager(persist_directory=args.index_dir)
    if not manager.load():
        print(f"No index at {args.index_dir}; a sync would build all "
              f"{len(report.documents)} documents.")
        return 0

    plan = plan_sync(report.documents, manager.known_ids())
    if args.no_prune:
        plan.retained, plan.removed = plan.removed, []
    print(f"Would sync: {plan.describe()}")
    _warn_on_total_churn(plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
