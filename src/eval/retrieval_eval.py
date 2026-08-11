"""Measured evaluation of retrieval, independent of the LLM.

PersonaAuditor grades a whole RAG cycle, which means a bad score there could be
retrieval's fault or generation's fault and the numbers cannot tell you which.
This module grades only the retrieval step: given a query, did the documents that
should have come back actually come back, and how far down the list were they?

That separation is the point. Every retrieval change worth making -- swapping the
embedding model, moving the flat index to IVF/HNSW, adding BM25 alongside the
dense search, changing chunk size or lore_weight -- trades recall for speed in
some ratio, and without a number here you cannot tell a speedup from a
regression. Nothing in this module calls Gemini, so it is cheap enough to run on
every index rebuild.

A case declares what *should* be retrieved as a list of matchers rather than a
single expected document, because more than one chunk of a 500-character-split
article can be the right answer and pinning one chunk id would make the suite
fail on every re-chunk.
"""

import json
from dataclasses import dataclass, field

from langchain_core.documents import Document

#: Shipped alongside this module rather than under data/, because it is part of
#: the test surface -- it must be present in a bare clone with no corpora.
DEFAULT_CASES_PATH = "src/eval/cases/retrieval.json"


@dataclass(frozen=True)
class Matcher:
    """One thing a case expects to see in the retrieved set.

    All populated fields must hold for the same document (AND), so
    {"speaker": "VADER", "phrase": "father"} means one Vader line containing
    "father" -- not a Vader line and, separately, something about fathers.
    """

    title: str | None = None
    speaker: str | None = None
    phrase: str | None = None

    def matches(self, doc: Document) -> bool:
        # An empty matcher would match every document and quietly inflate every
        # score, so treat it as a malformed case rather than a wildcard.
        if self.title is None and self.speaker is None and self.phrase is None:
            raise ValueError("Matcher has no criteria; it would match any document")

        if self.title is not None and (
            self.title.lower() not in (doc.metadata.get("title") or "").lower()
        ):
            return False
        # Exact after normalizing, not a substring: VADER must not match a line
        # credited to DARTH VADER, which is a different speaker string in the
        # corpora and is what PersonaManager.aliases exists to reconcile.
        if self.speaker is not None and (
            self.speaker.upper() != (doc.metadata.get("character") or "").upper().strip()
        ):
            return False
        return not (
            self.phrase is not None
            and self.phrase.lower() not in doc.page_content.lower()
        )


#: Which half of the index a case depends on. Recorded because the two halves
#: fail independently and for different reasons: the dialogue corpora are pinned
#: submodules, while data/raw/lore.json is whatever the last scrape produced --
#: Fandom rate-limits, so pages go missing without the scrape reporting failure.
#: A runner can therefore say "5 lore cases skipped, no lore in the index"
#: instead of scoring an absent corpus as a retrieval regression.
CORPORA = ("lore", "dialogue", "mixed")


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    character: str | None = None
    corpus: str = "mixed"
    expect: tuple[Matcher, ...] = field(default_factory=tuple)
    note: str = ""


def load_cases(path: str = DEFAULT_CASES_PATH) -> list[EvalCase]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    cases = []
    seen: set[str] = set()
    for entry in raw["cases"]:
        case_id = entry["id"]
        if case_id in seen:
            raise ValueError(f"Duplicate eval case id: {case_id}")
        seen.add(case_id)

        expect = tuple(Matcher(**matcher) for matcher in entry["expect"])
        if not expect:
            raise ValueError(f"Case {case_id} expects nothing, so it cannot fail")

        corpus = entry.get("corpus", "mixed")
        if corpus not in CORPORA:
            raise ValueError(
                f"Case {case_id} declares corpus {corpus!r}; expected one of {CORPORA}"
            )

        cases.append(
            EvalCase(
                id=case_id,
                query=entry["query"],
                character=entry.get("character"),
                corpus=corpus,
                expect=expect,
                note=entry.get("note", ""),
            )
        )
    return cases


def score_case(case: EvalCase, docs: list[Document]) -> dict:
    """Score one case against the documents retrieved for it.

    recall  -- fraction of the case's matchers hit by at least one document.
    rank    -- 1-based position of the first document hitting any matcher.
    rr      -- 1/rank, or 0.0 when nothing matched (the MRR convention).
    """
    hit_matchers = {
        index
        for index, matcher in enumerate(case.expect)
        for doc in docs
        if matcher.matches(doc)
    }

    rank = next(
        (
            position
            for position, doc in enumerate(docs, start=1)
            if any(matcher.matches(doc) for matcher in case.expect)
        ),
        None,
    )

    return {
        "id": case.id,
        "query": case.query,
        "character": case.character,
        "corpus": case.corpus,
        "retrieved": len(docs),
        "expected": len(case.expect),
        "matched": len(hit_matchers),
        "recall": len(hit_matchers) / len(case.expect),
        "hit": bool(hit_matchers),
        "rank": rank,
        "rr": 1.0 / rank if rank else 0.0,
        # Which matchers missed, so a failing case says what it wanted rather
        # than only that it scored 0.4.
        "missed": [
            {
                key: value
                for key, value in vars(matcher).items()
                if value is not None
            }
            for index, matcher in enumerate(case.expect)
            if index not in hit_matchers
        ],
    }


def run_suite(retriever, cases: list[EvalCase], k: int = 4) -> list[dict]:
    """Score every case. The retriever needs only a retrieve(query, character, k)."""
    return [
        score_case(case, retriever.retrieve(case.query, character=case.character, k=k))
        for case in cases
    ]


def summarize(results: list[dict], k: int | None = None) -> dict:
    """Collapse per-case scores into the numbers worth tracking over time."""
    total = len(results)
    if not total:
        return {"cases": 0}

    return {
        "cases": total,
        "k": k,
        # Mean per-case recall, i.e. macro-averaged: a case expecting four
        # documents does not outweigh one expecting a single document.
        "recall_at_k": sum(r["recall"] for r in results) / total,
        "hit_rate": sum(r["hit"] for r in results) / total,
        "mrr": sum(r["rr"] for r in results) / total,
        "empty_retrievals": sum(1 for r in results if r["retrieved"] == 0),
        "failures": [r["id"] for r in results if not r["hit"]],
        # Broken out because a whole-suite hit_rate hides the failure mode this
        # project actually has: an index built with no dialogue corpus scores
        # ~50% overall, which looks like mediocre retrieval rather than a
        # missing submodule.
        "by_corpus": {
            corpus: {
                "cases": len(group),
                "hit_rate": sum(r["hit"] for r in group) / len(group),
                "recall_at_k": sum(r["recall"] for r in group) / len(group),
            }
            for corpus in sorted({r["corpus"] for r in results})
            if (group := [r for r in results if r["corpus"] == corpus])
        },
    }


def check_thresholds(summary: dict, *, min_hit_rate: float, min_mrr: float) -> list[str]:
    """Return one message per breached floor; an empty list means it passed.

    Returns rather than raises so a caller can report every breach at once
    instead of only the first, and so a non-blocking CI report can print the
    messages without failing the job.
    """
    breaches = []
    if summary.get("cases", 0) == 0:
        return ["Suite ran zero cases -- there is nothing to be confident about."]

    if summary["hit_rate"] < min_hit_rate:
        breaches.append(
            f"hit_rate {summary['hit_rate']:.3f} < {min_hit_rate:.3f} "
            f"(missed: {', '.join(summary['failures']) or 'none'})"
        )
    if summary["mrr"] < min_mrr:
        breaches.append(f"mrr {summary['mrr']:.3f} < {min_mrr:.3f}")
    return breaches


def format_report(results: list[dict], summary: dict) -> str:
    """Plain-text report, one line per case then the totals."""
    lines = [f"{'case':<28} {'hit':<5} {'recall':<7} {'rank':<5} missed"]
    for result in results:
        missed = "; ".join(
            ",".join(f"{key}={value}" for key, value in matcher.items())
            for matcher in result["missed"]
        )
        rank = str(result["rank"]) if result["rank"] else "-"
        lines.append(
            f"{result['id']:<28} {'yes' if result['hit'] else 'NO':<5} "
            f"{result['recall']:<7.2f} {rank:<5} {missed}"
        )

    lines.append("")
    for corpus, stats in summary.get("by_corpus", {}).items():
        lines.append(
            f"  {corpus:<10} cases={stats['cases']:<3} "
            f"hit_rate={stats['hit_rate']:.3f} recall@k={stats['recall_at_k']:.3f}"
        )
    lines.append(
        f"cases={summary['cases']} k={summary.get('k')} "
        f"recall@k={summary['recall_at_k']:.3f} "
        f"hit_rate={summary['hit_rate']:.3f} mrr={summary['mrr']:.3f} "
        f"empty={summary['empty_retrievals']}"
    )
    return "\n".join(lines)
