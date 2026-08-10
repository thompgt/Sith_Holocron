import time
from typing import TYPE_CHECKING, List, Optional
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStoreManager
from src.observability import metrics

if TYPE_CHECKING:
    from src.llm.persona_manager import PersonaManager

class HybridRetriever:
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        persona_manager: Optional["PersonaManager"] = None,
    ):
        self.vs_manager = vector_store_manager
        # Used only to resolve a persona key to the speaker names it answers to
        # in the corpora. Constructed here when not supplied so a bare
        # HybridRetriever(vs) still filters correctly.
        if persona_manager is None:
            from src.llm.persona_manager import PersonaManager

            persona_manager = PersonaManager()
        self.pm = persona_manager

    def retrieve(
        self,
        query: str,
        character: Optional[str] = None,
        k: int = 4,
        lore_weight: float = 0.5
    ) -> List[Document]:
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
