import time
from collections import Counter
from typing import TYPE_CHECKING, Optional

from langchain_core.documents import Document

from src.observability import metrics
from src.retrieval.vector_store import VectorStoreManager

if TYPE_CHECKING:
    from src.llm.persona_manager import PersonaManager
    from src.retrieval.keyword_index import KeywordIndex

#: Reciprocal rank fusion constant. 60 is the value from the original RRF paper
#: and the usual default: large enough that the top few ranks are close together
#: (so a rank-1 hit in one retriever cannot outvote agreement across both), small
#: enough that rank 1 still clearly beats rank 20.
RRF_K = 60


def document_identity(document: Document):
    """What counts as "the same document" across two retrievers.

    Content plus the metadata that distinguishes otherwise-identical text, not
    object identity: the same chunk arrives as two distinct Document instances
    from the dense and lexical sides, and comparing objects would rank it twice
    instead of rewarding the agreement.
    """
    return (
        document.page_content,
        document.metadata.get("character"),
        document.metadata.get("title"),
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], key
) -> list[Document]:
    """Merge ranked lists by summing 1/(RRF_K + rank) across them.

    Rank-based rather than score-based on purpose: a cosine similarity and a
    BM25 score are not on the same scale, have no shared upper bound, and shift
    with corpus size, so any fixed weighting of the two raw numbers is
    arbitrary. Ranks are comparable by construction.

    The practical effect is that a document both retrievers rank modestly can
    outrank one that only a single retriever loves -- which is the entire reason
    to run two retrievers.
    """
    scores: dict = {}
    documents: dict = {}
    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            identity = key(document)
            documents.setdefault(identity, document)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (RRF_K + rank)

    # Insertion order breaks ties, so the dense retriever's ordering wins when
    # fusion is genuinely indifferent -- deterministic, and the safer default
    # given the lexical side has no notion of semantic similarity at all.
    return [
        documents[identity]
        for identity in sorted(scores, key=lambda i: -scores[i])
    ]


class HybridRetriever:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        persona_manager: Optional["PersonaManager"] = None,
        keyword_index: "KeywordIndex | None" = None,
        use_keyword: bool = True,
    ):
        self.vs_manager = vector_store_manager
        # Lexical retrieval is opt-out rather than opt-in so the eval, the API
        # and the CLI all get the same retriever without each remembering to
        # enable it. use_keyword=False exists to measure the difference.
        self.use_keyword = use_keyword
        self._keyword_index = keyword_index
        # Used only to resolve a persona key to the speaker names it answers to
        # in the corpora. Constructed here when not supplied so a bare
        # HybridRetriever(vs) still filters correctly.
        if persona_manager is None:
            from src.llm.persona_manager import PersonaManager

            persona_manager = PersonaManager()
        self.pm = persona_manager

    @property
    def keyword_index(self):
        """The lexical index, built from the loaded FAISS index on first use.

        Lazy because constructing it walks every document in the index, and the
        API builds its retriever during startup where that cost is paid before
        the first request rather than during it. Returns None when there is
        nothing loaded to build from -- the dense path still works, so a missing
        lexical index degrades retrieval rather than breaking it.
        """
        if self._keyword_index is None and self.use_keyword:
            from src.retrieval.keyword_index import KeywordIndex

            documents = self.vs_manager.all_documents()
            if not documents:
                return None
            self._keyword_index = KeywordIndex(documents)
        return self._keyword_index

    def retrieve(
        self,
        query: str,
        character: str | None = None,
        k: int = 4,
        lore_weight: float = 0.5
    ) -> list[Document]:
        """
        Retrieves documents and attempts to balance Lore vs. Dialogue.
        If a character is provided, it filters dialogue for that character.
        """
        persona_label = metrics.normalize_persona(character)
        started = time.perf_counter()
        allowed_speakers = self.pm.aliases(character)

        try:
            # 1. Fetch more than k to allow for filtering and re-ranking
            raw_results = self.vs_manager.search(query, k=k*3)

            # 1b. Fuse in lexical hits. The dense side is weak on the exact
            # proper nouns a lore corpus is made of -- the retrieval eval
            # measured a paraphrase of an indexed passage missing entirely while
            # a "Darth Bane" query hit at rank 1. Both lists are over-fetched to
            # k*3 so fusion has room to reorder before the balance step trims.
            keyword_index = self.keyword_index
            origins: dict = {}
            if keyword_index is not None:
                lexical = keyword_index.search(query, k=k * 3)
                for label, ranked in (("dense", raw_results), ("lexical", lexical)):
                    for document in ranked:
                        origins.setdefault(document_identity(document), set()).add(label)
                raw_results = reciprocal_rank_fusion(
                    [raw_results, lexical], key=document_identity
                )

            metrics.RETRIEVAL_CANDIDATES.labels(persona=persona_label).inc(len(raw_results))

            lore_docs = []
            dialogue_docs = []

            for doc in raw_results:
                doc_type = doc.metadata.get("type", "lore")

                if doc_type == "dialogue":
                    # If a character filter is active, keep only lines spoken by
                    # a name this persona answers to. Membership, not equality:
                    # the corpora credit Palpatine as DARTH SIDIOUS / EMPEROR,
                    # never as the persona key.
                    if allowed_speakers is not None:
                        speaker = (doc.metadata.get("character") or "").upper().strip()
                        if speaker in allowed_speakers:
                            dialogue_docs.append(doc)
                    else:
                        dialogue_docs.append(doc)
                else:
                    lore_docs.append(doc)

            # 2. Balance the results
            # We want a mix. Let's try to get k/2 from each if possible.
            target_lore = int(k * lore_weight)
            target_dialogue = k - target_lore

            final_docs = lore_docs[:target_lore] + dialogue_docs[:target_dialogue]

            # If we don't have enough of one, fill from whatever is left over.
            # Written as two sequential ifs rather than if/elif. Today the elif
            # was equivalent -- the two targets sum to exactly k, so a shortfall
            # implies the pool that fell short has no surplus, and at most one
            # branch can ever contribute. That equivalence is a property of the
            # target arithmetic, not of the backfill, and it silently stops
            # holding the moment the targets are computed any other way.
            if len(final_docs) < k:
                remaining = k - len(final_docs)
                if remaining > 0 and len(lore_docs) > target_lore:
                    extra = lore_docs[target_lore:target_lore + remaining]
                    final_docs.extend(extra)
                    remaining -= len(extra)
                if remaining > 0 and len(dialogue_docs) > target_dialogue:
                    final_docs.extend(dialogue_docs[target_dialogue:target_dialogue + remaining])

            result = final_docs[:k]
        except Exception as e:
            metrics.RETRIEVAL_ERRORS.labels(
                persona=persona_label, error_type=type(e).__name__
            ).inc()
            metrics.RETRIEVAL_DURATION.labels(persona=persona_label).observe(
                time.perf_counter() - started
            )
            raise

        metrics.RETRIEVAL_DURATION.labels(persona=persona_label).observe(
            time.perf_counter() - started
        )
        metrics.RETRIEVAL_RETURNED_DOCUMENTS.labels(persona=persona_label).observe(
            len(result)
        )

        # Attribute each *returned* document to the retriever(s) that surfaced
        # it. Counted from the returned set rather than the candidate pools for
        # the same reason as the type split below: this is what the LLM saw.
        # Empty when fusion did not run, so the series simply stops rather than
        # reporting everything as dense-only and looking like a working system.
        if origins:
            attribution = Counter(
                "both" if len(origins.get(document_identity(doc), ())) > 1
                else next(iter(origins.get(document_identity(doc), ("dense",))))
                for doc in result
            )
            for retriever, count in attribution.items():
                metrics.RETRIEVAL_SOURCE.labels(
                    persona=persona_label, retriever=retriever
                ).inc(count)

        # Counted from the *returned* set rather than the pre-rebalance pools, so
        # the lore:dialogue ratio on this counter is what the LLM actually saw --
        # that is the observable proof that the lore_weight rebalance (and the
        # backfill when one pool is short) is doing what it claims.
        lore_returned = sum(
            1 for doc in result if doc.metadata.get("type", "lore") != "dialogue"
        )
        if lore_returned:
            metrics.RETRIEVAL_DOCUMENTS_BY_TYPE.labels(
                persona=persona_label, source_type="lore"
            ).inc(lore_returned)
        dialogue_returned = len(result) - lore_returned
        if dialogue_returned:
            metrics.RETRIEVAL_DOCUMENTS_BY_TYPE.labels(
                persona=persona_label, source_type="dialogue"
            ).inc(dialogue_returned)

        return result
