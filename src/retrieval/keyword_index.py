"""BM25 keyword search over the same documents as the dense index.

The retrieval eval said to build this. Its `lore-rule-of-two-bane` case, whose
expectation is the literal string "Darth Bane", hits at rank 1, while
`lore-sith-code-passion` -- a paraphrase of a passage that is *in* the corpus --
misses entirely. That asymmetry is the standard weakness of dense-only
retrieval: embeddings capture topic, not tokens, and a lore corpus is mostly
proper nouns. A larger embedding model does not fix it; a lexical scorer does.

Implemented here rather than pulled in, because BM25 is a hundred lines with a
published formula, and the alternative (rank_bm25) is an unmaintained dependency
that would still need its own tokenizer decisions made here. The tokenizer is
the part that matters for this corpus, and it is visible and testable.

Scale note: this is an in-memory posting list rebuilt at load. Fine for the
~1.9k chunks here and for a few hundred thousand; past that it wants to live in
whatever serves the dense index rather than in the API process.
"""

import math
import re
from collections import Counter

from langchain_core.documents import Document

#: Words carrying no discriminative signal in *this* corpus. Deliberately short:
#: an aggressive stoplist is how a lexical index loses "The Force", and "no" is
#: load-bearing in a line like "There is no emotion".
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "they", "we", "what", "which", "who",
})

#: Standard BM25 parameters. k1 damps term-frequency saturation; b controls
#: length normalization.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords dropped.

    Apostrophes are kept inside tokens so "don't" stays one token rather than
    becoming "don" + "t", and hyphenated names split -- C-3PO indexes as "c" and
    "3po", which is what makes a query for "C-3PO" and one for "C3PO" both work.
    """
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class KeywordIndex:
    """An in-memory BM25 index over a fixed document set."""

    def __init__(self, documents: list[Document]):
        self.documents = list(documents)
        self._lengths: list[int] = []
        # term -> {document position: term frequency}
        self._postings: dict[str, dict[int, int]] = {}

        for position, document in enumerate(self.documents):
            tokens = tokenize(self._searchable_text(document))
            self._lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, {})[position] = count

        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    @staticmethod
    def _searchable_text(document: Document) -> str:
        """Body text plus the metadata worth matching on.

        The title is included because a lore chunk from the middle of an article
        often never repeats the article's own name, and "Rule of Two" is exactly
        the kind of query a lexical index should nail. The speaker is included so
        "what did Vader say about the Force" has a lexical path to Vader's lines.
        """
        parts = [document.page_content]
        for key in ("title", "character"):
            value = document.metadata.get(key)
            if value:
                parts.append(str(value))
        return " ".join(parts)

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative.

        Without the outer +1, a term appearing in more than half the documents
        gets a negative weight, and a query containing it would actively push
        matching documents down the ranking.
        """
        document_frequency = len(self._postings.get(term, ()))
        total = len(self.documents)
        return math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

    def search(self, query: str, k: int = 4, predicate=None) -> list[Document]:
        """Top-k documents by BM25 score. Empty when no query term is known."""
        return [
            document
            for document, _ in self.search_with_scores(query, k=k, predicate=predicate)
        ]

    def search_with_scores(
        self, query: str, k: int = 4, predicate=None
    ) -> list[tuple[Document, float]]:
        """`predicate` filters by metadata *before* the top-k cut.

        Filtering after the cut would be a different and much worse operation:
        a persona holding 2% of the corpus would have its lines squeezed out of
        the top-k by everyone else's before the filter ever ran.
        """
        if not self.documents:
            return []

        allowed = (
            None
            if predicate is None
            else {
                position
                for position, document in enumerate(self.documents)
                if predicate(document.metadata)
            }
        )

        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for position, frequency in postings.items():
                if allowed is not None and position not in allowed:
                    continue
                length_ratio = (
                    self._lengths[position] / self._average_length
                    if self._average_length
                    else 1.0
                )
                denominator = frequency + K1 * (1 - B + B * length_ratio)
                scores[position] = scores.get(position, 0.0) + idf * (
                    frequency * (K1 + 1) / denominator
                )

        # Ties broken by position so the ranking is deterministic; without it,
        # two equally-scored chunks could swap between runs and the eval's
        # rank-based metrics would flap for no reason.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.documents[position], score) for position, score in ranked[:k]]
