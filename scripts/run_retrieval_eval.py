"""Run the retrieval eval suite against a real index.

    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --k 8 --json data/eval/latest.json
    python scripts/run_retrieval_eval.py --min-hit-rate 0.8 --min-mrr 0.5

No Gemini call happens here, so no API key is needed -- only a built index
(python -m src.main builds one on first run). The scoring logic itself is unit
tested in tests/test_retrieval_eval.py against a stub retriever; this script is
the part that needs real embeddings and is therefore not in CI.

Exit codes:
  0  suite ran and any thresholds given were met
  1  a threshold was breached
  2  the suite could not run at all (no index, or no cases)

The distinction in 1 vs 2 matters: "retrieval got worse" and "there was nothing
to measure" are different problems and this project has shipped both.
"""

import argparse
import json
import os
import sys

# Importable as a script from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.retrieval_eval import (  # noqa: E402
    DEFAULT_CASES_PATH,
    check_thresholds,
    format_report,
    load_cases,
    run_suite,
    summarize,
)
from src.llm.persona_manager import PersonaManager  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.vector_store import IndexTrustError, VectorStoreManager  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--index-dir", default="data/vector_store")
    parser.add_argument("--k", type=int, default=4, help="documents per query (default 4)")
    parser.add_argument(
        "--json",
        dest="json_path",
        help="also write the full per-case results here, for diffing runs",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        help="exit 1 if the fraction of cases retrieving anything expected falls below this",
    )
    parser.add_argument("--min-mrr", type=float, help="exit 1 if mean reciprocal rank falls below this")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Could not load cases from {args.cases}: {exc}", file=sys.stderr)
        return 2

    vs_manager = VectorStoreManager(persist_directory=args.index_dir)
    try:
        loaded = vs_manager.load()
    except IndexTrustError as exc:
        # Do not paper over this by setting the trust override here. An index
        # that fails verification is exactly the situation the check exists for.
        print(f"Index at {args.index_dir} failed verification:\n{exc}", file=sys.stderr)
        return 2

    if not loaded:
        print(
            f"No index at {args.index_dir}. Build one with `python -m src.main` "
            f"(and `python scripts/fetch_corpora.py` first, or the dialogue "
            f"cases will all miss for want of a corpus).",
            file=sys.stderr,
        )
        return 2

    retriever = HybridRetriever(vs_manager, persona_manager=PersonaManager())
    results = run_suite(retriever, cases, k=args.k)
    summary = summarize(results, k=args.k)

    print(format_report(results, summary))

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or ".", exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "results": results}, handle, indent=2)
        print(f"\nWrote {args.json_path}")

    # Only enforce the floors the caller actually asked for. Baking in defaults
    # would mean the first honest run of a new case set fails the build for
    # telling the truth about where retrieval stands.
    if args.min_hit_rate is None and args.min_mrr is None:
        return 0

    breaches = check_thresholds(
        summary,
        min_hit_rate=args.min_hit_rate if args.min_hit_rate is not None else 0.0,
        min_mrr=args.min_mrr if args.min_mrr is not None else 0.0,
    )
    if breaches:
        print("\nThresholds breached:", file=sys.stderr)
        for breach in breaches:
            print(f"  - {breach}", file=sys.stderr)
        return 1

    print("\nThresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
