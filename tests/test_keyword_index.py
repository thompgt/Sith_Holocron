"""Tests for the BM25 keyword index.

These assert ranking *properties* -- that a rarer term outranks a common one,
that a shorter document wins a tie, that a proper noun beats a paraphrase --
rather than exact scores. Pinning float scores would turn any parameter tune
into a test rewrite while catching nothing a property assertion misses.
"""

import pytest
from langchain_core.documents import Document

from src.retrieval.keyword_index import KeywordIndex, tokenize


def doc(text, **metadata):
    return Document(page_content=text, metadata=metadata)


# --- tokenizer -------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Rule of Two") == ["rule", "two"]


def test_tokenize_keeps_apostrophes_inside_words():
    # "don't" must not become "don" + "t"; the fragments match nothing useful.
    assert tokenize("Don't underestimate") == ["don't", "underestimate"]


def test_tokenize_splits_hyphenated_names():
    """C-3PO indexes as c + 3po, so "C-3PO" and "C3PO" have tokens in common."""
    assert tokenize("C-3PO") == ["c", "3po"]


def test_negations_survive_tokenization():
    """"There is no emotion" loses its meaning if "no" is a stopword."""
    assert "no" in tokenize("There is no emotion, there is peace")


def test_tokenize_ignores_punctuation_only_input():
    assert tokenize("... --- !!!") == []


# --- ranking ---------------------------------------------------------------


def test_exact_proper_noun_outranks_a_topical_paraphrase():
    """The case the eval flagged: dense retrieval loses on proper nouns."""
    index = KeywordIndex(
        [
            doc("A decree about masters and apprentices in general."),
            doc("The Rule of Two was established by Darth Bane."),
        ]
    )

    assert index.search("Darth Bane", k=1)[0].page_content.endswith("Darth Bane.")


def test_a_rare_term_outweighs_a_common_one():
    common = [doc("the sith are powerful") for _ in range(20)]
    rare = doc("the sith revere Bane")
    index = KeywordIndex([*common, rare])

    # "sith" appears everywhere and should barely discriminate; "bane" is unique.
    assert index.search("sith bane", k=1)[0] is rare


def test_shorter_documents_win_when_term_frequency_ties():
    """Length normalization: one hit in ten words beats one hit in a hundred."""
    short = doc("Bane decreed it.")
    long = doc("Bane decreed it. " + "Filler about unrelated galactic matters. " * 20)
    index = KeywordIndex([long, short])

    assert index.search("Bane", k=1)[0] is short


def test_repeated_terms_score_higher_but_with_diminishing_returns():
    once = doc("Bane " + "filler " * 10)
    thrice = doc("Bane Bane Bane " + "filler " * 10)
    index = KeywordIndex([once, thrice])

    scores = dict(
        (document.page_content, score)
        for document, score in index.search_with_scores("Bane", k=2)
    )
    ratio = scores[thrice.page_content] / scores[once.page_content]

    assert ratio > 1  # more occurrences is a stronger match...
    assert ratio < 3  # ...but not three times as strong


def test_idf_never_goes_negative_for_a_term_in_most_documents():
    """A negative weight would push matching documents *down* the ranking."""
    index = KeywordIndex([doc("sith power") for _ in range(10)] + [doc("jedi")])

    scores = [score for _, score in index.search_with_scores("sith", k=11)]

    assert scores
    assert all(score > 0 for score in scores)


# --- metadata --------------------------------------------------------------


def test_title_is_searchable_even_when_absent_from_the_body():
    """A mid-article chunk rarely repeats the article's own name."""
    index = KeywordIndex(
        [
            doc("Unrelated passage about droids."),
            doc("The apprentice must kill the master.", title="Rule of Two"),
        ]
    )

    assert index.search("Rule of Two", k=1)[0].metadata["title"] == "Rule of Two"


def test_speaker_is_searchable():
    index = KeywordIndex(
        [
            doc("I find your lack of faith disturbing.", character="VADER"),
            doc("You are a member of the Rebel Alliance.", character="LEIA"),
        ]
    )

    assert index.search("Vader faith", k=1)[0].metadata["character"] == "VADER"


# --- degenerate inputs -----------------------------------------------------


def test_query_with_no_known_terms_returns_nothing():
    """Must return empty, not an arbitrary top-k -- silent wrong answers."""
    index = KeywordIndex([doc("Bane decreed it")])

    assert index.search("xyzzy plugh", k=4) == []


def test_query_of_only_stopwords_returns_nothing():
    index = KeywordIndex([doc("Bane decreed it")])

    assert index.search("the and of", k=4) == []


def test_empty_index_searches_cleanly():
    assert KeywordIndex([]).search("anything", k=4) == []


def test_k_caps_the_result_count():
    index = KeywordIndex([doc(f"Bane decree number {n}") for n in range(10)])

    assert len(index.search("Bane", k=3)) == 3


def test_ranking_is_deterministic_across_identical_indexes():
    """Tie-broken by position, so eval rank metrics do not flap between runs."""
    documents = [doc("Bane decreed it") for _ in range(5)]
    first = KeywordIndex(documents).search_with_scores("Bane", k=5)
    second = KeywordIndex(documents).search_with_scores("Bane", k=5)

    assert [score for _, score in first] == [score for _, score in second]
    assert [d.page_content for d, _ in first] == [d.page_content for d, _ in second]


def test_scores_are_returned_in_descending_order():
    index = KeywordIndex(
        [doc("filler " * 30 + "Bane"), doc("Bane"), doc("Bane Bane")]
    )
    scores = [score for _, score in index.search_with_scores("Bane", k=3)]

    assert scores == sorted(scores, reverse=True)


def test_documents_with_no_indexable_tokens_do_not_break_averaging():
    """A punctuation-only chunk has length 0 and must not divide by zero."""
    index = KeywordIndex([doc("---"), doc("Bane decreed it")])

    assert index.search("Bane", k=1)[0].page_content == "Bane decreed it"


def test_search_with_scores_and_search_agree():
    index = KeywordIndex([doc("Bane"), doc("Sidious"), doc("Bane and Sidious")])

    assert index.search("Bane", k=2) == [
        document for document, _ in index.search_with_scores("Bane", k=2)
    ]


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_queries_return_nothing(query):
    assert KeywordIndex([doc("Bane")]).search(query, k=4) == []
