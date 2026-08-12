import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever


class DenseOnlyStore:
    """Base for stubs that exercise the dense path in isolation.

    all_documents() returning nothing is what switches the lexical half off:
    HybridRetriever builds its BM25 index from the loaded FAISS index, so a
    store with no documents to enumerate has no lexical index to build. The
    tests below are about the balance/filter/backfill logic, and fusing a second
    ranker into them would change what they measure. Fusion has its own tests.

    search() honours `filter` because the real one does, and a double that
    ignored it would let a retriever that never filters pass every test here.
    Ranking is not modelled -- these stubs return a fixed set regardless of the
    query, which is what makes the assertions about *selection* unambiguous.
    """

    def documents(self):
        raise NotImplementedError

    def search(self, query, k=4, filter=None):
        documents = self.documents()
        if filter is not None:
            documents = [d for d in documents if filter(d.metadata)]
        return documents[:k]

    def all_documents(self):
        return []


class MockVectorStore(DenseOnlyStore):
    def documents(self):
        # A mix of lore and dialogue
        return [
            Document(page_content="Fact about Sith.", metadata={"type": "lore", "title": "History"}),
            Document(page_content="Dialogue from Vader.", metadata={"type": "dialogue", "character": "VADER"})
        ]

@pytest.fixture
def hybrid_retriever():
    mock_vs = MockVectorStore()
    return HybridRetriever(vector_store_manager=mock_vs)

def test_hybrid_retrieval_balance(hybrid_retriever):
    # Test that it fetches results
    results = hybrid_retriever.retrieve("Sith history", k=2)
    assert len(results) == 2

    # Check if we have both types (or at least the structure is respected)
    types = [doc.metadata.get("type") for doc in results]
    assert "lore" in types or "dialogue" in types

def test_persona_filtering_logic(hybrid_retriever):
    # If we filter by VADER, we should prioritize or only see VADER dialogue
    results = hybrid_retriever.retrieve("plans", character="VADER")
    for doc in results:
        if doc.metadata.get("type") == "dialogue":
            assert doc.metadata.get("character") == "VADER"


class AliasVectorStore(DenseOnlyStore):
    """Dialogue credited the way the corpora actually credit it."""

    def __init__(self, speakers):
        self.speakers = speakers

    def documents(self):
        return [
            Document(
                page_content=f"Line by {s}.",
                metadata={"type": "dialogue", "character": s},
            )
            for s in self.speakers
        ]


def _speakers(docs):
    return {d.metadata["character"] for d in docs if d.metadata.get("type") == "dialogue"}


def test_palpatine_matches_his_script_credits():
    # The prequels credit him DARTH SIDIOUS and the OT credits him EMPEROR;
    # nothing in either corpus is attributed to the persona key "PALPATINE"
    # alone. Matching the key by equality returned zero dialogue documents.
    vs = AliasVectorStore(["DARTH SIDIOUS", "EMPEROR", "LUKE"])
    retriever = HybridRetriever(vector_store_manager=vs)

    docs = retriever.retrieve("power", character="PALPATINE", k=4)

    assert _speakers(docs) == {"DARTH SIDIOUS", "EMPEROR"}


def test_generic_sith_matches_any_dark_side_speaker():
    # No line is ever spoken by a character named GENERIC_SITH, so this persona
    # could never match anything at all.
    vs = AliasVectorStore(["DARTH MAUL", "VADER", "OBI-WAN"])
    retriever = HybridRetriever(vector_store_manager=vs)

    docs = retriever.retrieve("the dark side", character="GENERIC_SITH", k=4)

    assert _speakers(docs) == {"DARTH MAUL", "VADER"}


def test_unknown_persona_falls_back_instead_of_matching_nothing():
    vs = AliasVectorStore(["VADER", "JAR JAR"])
    retriever = HybridRetriever(vector_store_manager=vs)

    docs = retriever.retrieve("anything", character="NOT_A_PERSONA", k=4)

    assert _speakers(docs) == {"VADER"}


class SplitVectorStore(DenseOnlyStore):
    """One lore surplus and one dialogue surplus, neither enough on its own."""

    def documents(self):
        return [
            Document(page_content="lore 1", metadata={"type": "lore"}),
            Document(page_content="lore 2", metadata={"type": "lore"}),
            Document(page_content="lore 3", metadata={"type": "lore"}),
            Document(
                page_content="line 1", metadata={"type": "dialogue", "character": "VADER"}
            ),
            Document(
                page_content="line 2", metadata={"type": "dialogue", "character": "VADER"}
            ),
            Document(
                page_content="line 3", metadata={"type": "dialogue", "character": "VADER"}
            ),
        ]


@pytest.mark.parametrize("lore_weight", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_backfill_returns_k_whenever_the_pools_can_cover_it(k, lore_weight):
    """Backfill must saturate k regardless of how the targets split.

    The store holds 3 lore and 3 dialogue, so every k up to 6 is coverable no
    matter which way lore_weight skews the targets. This pins the property the
    backfill exists to provide -- a short pool gets topped up from the other one.
    """
    retriever = HybridRetriever(vector_store_manager=SplitVectorStore())

    docs = retriever.retrieve("q", character="VADER", k=k, lore_weight=lore_weight)

    assert len(docs) == k


# --- lexical fusion --------------------------------------------------------


class FusionStore:
    """A store whose dense ranking is deliberately wrong for the query.

    The dense side returns filler first and the answer last; the lexical side,
    built over the same documents, should pull the answer forward. That is the
    exact shape of the failure the retrieval eval measured -- a paraphrase-shaped
    query missing a passage whose own words are in the corpus.
    """

    ANSWER = Document(
        page_content="The Rule of Two was established by Darth Bane.",
        metadata={"type": "lore", "title": "Rule of Two"},
    )
    FILLER = [
        Document(page_content=f"Unrelated passage {n} about droids.", metadata={"type": "lore"})
        for n in range(6)
    ]

    def search(self, query, k=4, filter=None):
        documents = [*self.FILLER, self.ANSWER]
        if filter is not None:
            documents = [d for d in documents if filter(d.metadata)]
        return documents[:k]

    def all_documents(self):
        return [*self.FILLER, self.ANSWER]


def test_lexical_fusion_pulls_an_exact_match_forward():
    retriever = HybridRetriever(vector_store_manager=FusionStore())

    docs = retriever.retrieve("Darth Bane", k=3)

    assert docs[0].page_content == FusionStore.ANSWER.page_content


def test_dense_only_retrieval_misses_it_without_fusion():
    """Pins that the test above is measuring fusion and not something else."""
    retriever = HybridRetriever(vector_store_manager=FusionStore(), use_keyword=False)

    docs = retriever.retrieve("Darth Bane", k=3)

    assert FusionStore.ANSWER.page_content not in [d.page_content for d in docs]


def test_fusion_does_not_duplicate_a_document_both_retrievers_return():
    retriever = HybridRetriever(vector_store_manager=FusionStore())

    docs = retriever.retrieve("Darth Bane droids", k=6)

    assert len(docs) == len({d.page_content for d in docs})


def test_retrieval_survives_a_store_with_nothing_to_enumerate():
    """A missing lexical index must degrade retrieval, not break it."""
    retriever = HybridRetriever(vector_store_manager=MockVectorStore())

    assert retriever.keyword_index is None
    assert retriever.retrieve("Sith", k=2)


def test_keyword_index_is_built_once_and_reused():
    store = FusionStore()
    retriever = HybridRetriever(vector_store_manager=store)

    assert retriever.keyword_index is retriever.keyword_index


def test_rrf_rewards_agreement_over_a_single_strong_opinion():
    """A document both rankers place well beats one only the first ranks first.

    The separation has to be real to show up: with RRF_K=60 the gap between
    rank 1 and rank 3 is under a thousandth, which is the point of that constant
    -- adjacent top ranks are near-ties, and agreement is what breaks them. So
    the favourite is ranked 1st and 10th while the agreed document is 2nd twice.

    The assertion is on their relative order rather than on first place: a filler
    document ranked 3rd and 1st scores fractionally above the agreed document,
    which is correct behaviour and not what this test is about.
    """
    from src.retrieval.hybrid_retriever import reciprocal_rank_fusion

    favourite = Document(page_content="only the dense side likes this")
    agreed = Document(page_content="both sides like this")
    filler = [Document(page_content=f"filler {n}") for n in range(8)]

    fused = reciprocal_rank_fusion(
        [
            [favourite, agreed, *filler],
            [filler[0], agreed, *filler[1:], favourite],
        ],
        key=lambda d: d.page_content,
    )
    order = [document.page_content for document in fused]

    assert order.index(agreed.page_content) < order.index(favourite.page_content)


def test_rrf_on_an_empty_second_list_preserves_the_first_ordering():
    from src.retrieval.hybrid_retriever import reciprocal_rank_fusion

    ranked = [Document(page_content=str(n)) for n in range(4)]
    fused = reciprocal_rank_fusion([ranked, []], key=lambda d: d.page_content)

    assert [d.page_content for d in fused] == [d.page_content for d in ranked]


def _source_counts(persona="other"):
    from src.observability import metrics

    return {
        retriever: metrics.RETRIEVAL_SOURCE.labels(
            persona=persona, retriever=retriever
        )._value.get()
        for retriever in ("dense", "lexical", "both")
    }


def test_returned_documents_are_attributed_to_a_retriever():
    """A broken lexical half is otherwise invisible.

    BM25 contributing nothing looks exactly like BM25 agreeing with the dense
    ranking, and both look like healthy retrieval everywhere else in the
    dashboard.
    """
    before = _source_counts()
    retriever = HybridRetriever(vector_store_manager=FusionStore())

    docs = retriever.retrieve("Darth Bane", k=3)

    after = _source_counts()
    recorded = sum(after[key] - before[key] for key in after)
    assert recorded == len(docs)
    # The answer is returned by both retrievers, so it must not be booked as
    # dense-only -- that is the number someone would read as "BM25 is idle".
    assert after["both"] > before["both"]


def test_no_attribution_is_recorded_when_fusion_did_not_run():
    """The series stops rather than reporting everything as dense-only."""
    before = _source_counts()
    retriever = HybridRetriever(vector_store_manager=MockVectorStore())

    retriever.retrieve("Sith", k=2)

    assert _source_counts() == before


# --- persona-scoped retrieval ----------------------------------------------


class RareSpeakerStore(DenseOnlyStore):
    """A persona who is a tiny minority of the corpus, as Vader really is.

    Vader speaks 41 of 1,908 chunks. A general top-k is filled by whoever else
    is talking, so filtering it afterwards leaves nothing -- which is exactly
    how the eval's most-quoted line in the corpus became unreachable.
    """

    TARGET = Document(
        page_content="I find your lack of faith disturbing.",
        metadata={"type": "dialogue", "character": "VADER"},
    )

    def documents(self):
        crowd = [
            Document(
                page_content=f"Line {n} by someone else.",
                metadata={"type": "dialogue", "character": "TARKIN"},
            )
            for n in range(30)
        ]
        return [*crowd, self.TARGET]


def test_a_rare_persona_is_retrieved_from_a_search_of_their_own():
    retriever = HybridRetriever(vector_store_manager=RareSpeakerStore())

    docs = retriever.retrieve("faith", character="VADER", k=4)

    assert RareSpeakerStore.TARGET.page_content in [d.page_content for d in docs]


def test_the_general_pool_alone_would_have_missed_it():
    """Pins that the test above measures scoped retrieval, not luck.

    The store returns the crowd first, so anything reading only the general
    top-k -- and filtering it afterwards -- cannot see the target at all.
    """
    store = RareSpeakerStore()

    general = store.search("faith", k=12)

    assert RareSpeakerStore.TARGET not in general


def test_scoped_retrieval_still_only_returns_the_personas_own_lines():
    retriever = HybridRetriever(vector_store_manager=RareSpeakerStore())

    docs = retriever.retrieve("faith", character="VADER", k=4)

    assert _speakers(docs) == {"VADER"}


def test_no_persona_filter_means_no_scoped_search():
    """Without a character there is nothing to scope to, and dialogue comes
    from the general pool as before."""
    retriever = HybridRetriever(vector_store_manager=RareSpeakerStore())

    docs = retriever.retrieve("faith", k=4)

    assert _speakers(docs) == {"TARKIN"}
