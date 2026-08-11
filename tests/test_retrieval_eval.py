"""Tests for the retrieval eval harness.

Deliberately index-free: every case here drives a stub retriever with
hand-built Documents. A harness whose own correctness depended on a 90 MB
embedding download and a built FAISS index could not run in CI, and a scoring
bug in it would be indistinguishable from the retrieval regressions it exists
to catch.

The live run against a real index is scripts/run_retrieval_eval.py.
"""

import json

import pytest
from langchain_core.documents import Document

from src.eval.retrieval_eval import (
    DEFAULT_CASES_PATH,
    EvalCase,
    Matcher,
    check_thresholds,
    format_report,
    load_cases,
    run_suite,
    score_case,
    summarize,
)


def lore(text, title="Rule of Two"):
    return Document(page_content=text, metadata={"type": "lore", "title": title})


def line(text, character="VADER"):
    return Document(
        page_content=text, metadata={"type": "dialogue", "character": character}
    )


class StubRetriever:
    """Returns a canned document list per query and records how it was called."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query, character=None, k=4):
        self.calls.append((query, character, k))
        return self.responses.get(query, [])[:k]


# --- matchers --------------------------------------------------------------


def test_title_match_is_substring_and_case_insensitive():
    assert Matcher(title="rule of two").matches(lore("x", title="Rule of Two"))
    assert not Matcher(title="Code of the Sith").matches(lore("x", title="Rule of Two"))


def test_speaker_match_is_exact_after_normalizing():
    # The corpora pad and case speaker names inconsistently, but a speaker match
    # must stay exact -- substring matching would make VADER match DARTH VADER's
    # lines and, worse, make "BEN" match "BEN KENOBI" but not vice versa.
    assert Matcher(speaker="vader").matches(line("x", character="  VADER "))
    assert not Matcher(speaker="VADER").matches(line("x", character="DARTH VADER"))


def test_matcher_fields_are_anded_not_ored():
    matcher = Matcher(speaker="VADER", phrase="father")
    assert matcher.matches(line("I am your father"))
    assert not matcher.matches(line("I am your father", character="LUKE"))
    assert not matcher.matches(line("The Force is strong"))


def test_missing_metadata_is_a_miss_not_a_crash():
    bare = Document(page_content="text", metadata={})
    assert not Matcher(title="Anything").matches(bare)
    assert not Matcher(speaker="VADER").matches(bare)
    assert Matcher(phrase="text").matches(bare)


def test_empty_matcher_is_rejected_rather_than_matching_everything():
    with pytest.raises(ValueError, match="no criteria"):
        Matcher().matches(lore("anything at all"))


# --- scoring ---------------------------------------------------------------


def test_perfect_case_scores_one():
    case = EvalCase(id="c", query="q", expect=(Matcher(phrase="passion"),))
    result = score_case(case, [lore("There is only Passion")])

    assert result["hit"] is True
    assert result["recall"] == 1.0
    assert result["rank"] == 1
    assert result["rr"] == 1.0
    assert result["missed"] == []


def test_partial_recall_reports_which_matcher_missed():
    case = EvalCase(
        id="c",
        query="q",
        expect=(Matcher(title="Rule of Two"), Matcher(speaker="VADER")),
    )
    result = score_case(case, [lore("Darth Bane decreed it")])

    assert result["hit"] is True  # something matched...
    assert result["recall"] == 0.5  # ...but not everything
    assert result["missed"] == [{"speaker": "VADER"}]


def test_rank_and_rr_use_the_first_matching_document():
    case = EvalCase(id="c", query="q", expect=(Matcher(speaker="VADER"),))
    docs = [lore("a"), lore("b"), line("I find your lack of faith disturbing")]
    result = score_case(case, docs)

    assert result["rank"] == 3
    assert result["rr"] == pytest.approx(1 / 3)


def test_total_miss_scores_zero_with_no_rank():
    case = EvalCase(id="c", query="q", expect=(Matcher(phrase="nowhere"),))
    result = score_case(case, [lore("something else")])

    assert result["hit"] is False
    assert result["recall"] == 0.0
    assert result["rank"] is None
    assert result["rr"] == 0.0


def test_empty_retrieval_scores_zero_and_is_counted():
    case = EvalCase(id="c", query="q", expect=(Matcher(phrase="anything"),))
    result = score_case(case, [])

    assert result["retrieved"] == 0
    assert result["hit"] is False
    assert summarize([result])["empty_retrievals"] == 1


# --- suite and summary -----------------------------------------------------


def test_run_suite_passes_character_and_k_through():
    cases = [
        EvalCase(id="a", query="q1", character="VADER", expect=(Matcher(phrase="x"),)),
        EvalCase(id="b", query="q2", expect=(Matcher(phrase="x"),)),
    ]
    retriever = StubRetriever({"q1": [line("x marks it")], "q2": [lore("no")]})

    results = run_suite(retriever, cases, k=7)

    assert retriever.calls == [("q1", "VADER", 7), ("q2", None, 7)]
    assert [r["hit"] for r in results] == [True, False]


def test_summary_recall_is_macro_averaged_not_pooled():
    # One case expects 4 matchers and hits 1; the other expects 1 and hits it.
    # Pooling matchers would give 2/5 = 0.4. Macro-averaging gives
    # (0.25 + 1.0) / 2 = 0.625 -- the case expecting more does not dominate.
    many = EvalCase(
        id="many",
        query="q",
        expect=tuple(Matcher(phrase=p) for p in ("a", "b", "c", "d")),
    )
    one = EvalCase(id="one", query="q", expect=(Matcher(phrase="a"),))
    results = [score_case(many, [lore("a")]), score_case(one, [lore("a")])]

    assert summarize(results)["recall_at_k"] == pytest.approx(0.625)


def test_summary_lists_failing_case_ids_and_mrr():
    hit = EvalCase(id="hit", query="q", expect=(Matcher(phrase="a"),))
    miss = EvalCase(id="miss", query="q", expect=(Matcher(phrase="z"),))
    summary = summarize([score_case(hit, [lore("a")]), score_case(miss, [lore("a")])], k=4)

    assert summary["cases"] == 2
    assert summary["k"] == 4
    assert summary["hit_rate"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["failures"] == ["miss"]


def test_by_corpus_separates_a_missing_half_of_the_index():
    """The whole reason by_corpus exists: distinguish bad retrieval from no data."""
    lore_case = EvalCase(id="l", query="q", corpus="lore", expect=(Matcher(phrase="a"),))
    dialogue_case = EvalCase(
        id="d", query="q", corpus="dialogue", expect=(Matcher(speaker="VADER"),)
    )
    # Lore retrieves fine; the dialogue corpus is simply not in the index.
    summary = summarize([score_case(lore_case, [lore("a")]), score_case(dialogue_case, [])])

    assert summary["hit_rate"] == 0.5  # looks like mediocre retrieval...
    assert summary["by_corpus"]["lore"]["hit_rate"] == 1.0  # ...but it is not
    assert summary["by_corpus"]["dialogue"]["hit_rate"] == 0.0


def test_summary_of_nothing_does_not_divide_by_zero():
    assert summarize([]) == {"cases": 0}


# --- thresholds ------------------------------------------------------------


def test_thresholds_pass_quietly():
    summary = summarize([score_case(EvalCase("c", "q", expect=(Matcher(phrase="a"),)), [lore("a")])])
    assert check_thresholds(summary, min_hit_rate=1.0, min_mrr=1.0) == []


def test_thresholds_report_every_breach_not_just_the_first():
    summary = summarize([score_case(EvalCase("c", "q", expect=(Matcher(phrase="z"),)), [lore("a")])])
    breaches = check_thresholds(summary, min_hit_rate=0.9, min_mrr=0.9)

    assert len(breaches) == 2
    assert any("hit_rate" in b for b in breaches)
    assert any("mrr" in b for b in breaches)


def test_an_empty_suite_is_a_breach_not_a_pass():
    """A suite that collected no cases must never read as green."""
    assert check_thresholds(summarize([]), min_hit_rate=0.0, min_mrr=0.0)


# --- the shipped case set --------------------------------------------------


def test_shipped_cases_load_and_are_well_formed():
    cases = load_cases()

    assert len(cases) >= 10
    assert len({c.id for c in cases}) == len(cases)
    for case in cases:
        assert case.query.strip()
        assert case.expect
        assert case.note, f"{case.id} has no note explaining what it guards"


def test_shipped_cases_cover_both_halves_of_the_index():
    corpora = {case.corpus for case in load_cases()}
    assert {"lore", "dialogue"} <= corpora


def test_shipped_cases_include_a_persona_filtered_case():
    assert any(case.character for case in load_cases())


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.json"
    entry = {"id": "same", "query": "q", "expect": [{"phrase": "a"}]}
    path.write_text(json.dumps({"cases": [entry, entry]}))

    with pytest.raises(ValueError, match="Duplicate eval case id"):
        load_cases(str(path))


def test_unknown_corpus_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {"cases": [{"id": "a", "query": "q", "corpus": "screenplay",
                        "expect": [{"phrase": "a"}]}]}
        )
    )

    with pytest.raises(ValueError, match="declares corpus"):
        load_cases(str(path))


def test_case_expecting_nothing_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"id": "a", "query": "q", "expect": []}]}))

    with pytest.raises(ValueError, match="cannot fail"):
        load_cases(str(path))


def test_default_cases_path_points_at_the_shipped_file():
    # Guards the harness against being wired to a path that does not exist,
    # which would make a live run report zero cases and look like a pass.
    load_cases(DEFAULT_CASES_PATH)


# --- report ----------------------------------------------------------------


def test_report_names_the_failing_case_and_what_it_wanted():
    miss = EvalCase(id="lore-missing", query="q", expect=(Matcher(title="Sith"),))
    results = [score_case(miss, [lore("unrelated", title="C-3PO")])]
    report = format_report(results, summarize(results, k=4))

    assert "lore-missing" in report
    assert "NO" in report
    assert "title=Sith" in report
