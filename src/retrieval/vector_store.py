import hashlib
import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Loading a FAISS index means unpickling its docstore, and unpickling is
# arbitrary code execution. The threat is not hypothetical for this project: the
# index is gitignored, so every user builds or obtains one out-of-band, and a
# swapped .pkl runs as whatever imports src.api.server -- at module scope,
# before any request is served.
#
# So an index is loaded only when it is provably the one this machine built: save()
# records a manifest of file digests, and load() re-verifies them. Setting
# HOLOCRON_TRUST_INDEX=1 skips verification, for the case where an index is
# deliberately taken from somewhere else and the operator accepts that risk.
TRUST_ENV_VAR = "HOLOCRON_TRUST_INDEX"
MANIFEST_SUFFIX = ".manifest.json"


class IndexTrustError(RuntimeError):
    """An index exists but could not be shown to be the one we built."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trust_override_enabled() -> bool:
    return os.getenv(TRUST_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


class VectorStoreManager:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "data/vector_store",
        index_name: str = "sith_holocron_index"
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = persist_directory
        self.index_name = index_name
        self.vector_store: FAISS | None = None

    def add_documents(self, documents: list[Document], ids: list[str] | None = None):
        """
        Adds documents to the vector store. Creates a new index if one doesn't exist.

        `ids` are FAISS docstore keys. Passing content-derived ids is what makes
        an incremental rebuild possible: the index then records which chunks it
        holds, so a later build can diff against it instead of re-embedding the
        whole corpus. See src.ingestion.pipeline.chunk_id.
        """
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings, ids=ids)
        else:
            self.vector_store.add_documents(documents, ids=ids)

    def all_documents(self) -> list[Document]:
        """Every document in the index, for building a secondary index over it.

        The lexical index in keyword_index.py is derived from this rather than
        re-read from the corpora, so the two halves of retrieval can never
        disagree about what is searchable -- a keyword hit on a chunk the dense
        index does not hold would be unreachable through the retriever anyway.
        """
        if self.vector_store is None:
            return []
        docstore = self.vector_store.docstore
        return [
            document
            for document in (
                docstore.search(doc_id)
                for doc_id in self.vector_store.index_to_docstore_id.values()
            )
            # InMemoryDocstore.search returns an error *string* for a missing id
            # rather than raising, so filter on type instead of truthiness.
            if isinstance(document, Document)
        ]

    def known_ids(self) -> set[str]:
        """Docstore keys currently in the index; empty when there is no index.

        Read from the index itself rather than a sidecar file, so the record of
        what is indexed cannot drift away from what is actually indexed.
        """
        if self.vector_store is None:
            return set()
        return set(self.vector_store.index_to_docstore_id.values())

    def delete(self, ids: list[str]) -> None:
        """Remove documents by docstore key. A no-op for an empty id list."""
        if ids and self.vector_store is not None:
            self.vector_store.delete(ids)

    def search(self, query: str, k: int = 4, filter=None) -> list[Document]:
        """
        Performs a similarity search in the vector store.

        `filter` is a callable over a Document's metadata. FAISS has no notion
        of a filtered index -- it fetches `fetch_k` nearest neighbours and then
        discards the ones that fail the predicate -- so fetch_k has to be large
        enough that k survivors remain. A persona is a small slice of this
        corpus (Vader speaks 41 of 1,908 chunks), and the default fetch_k of 20
        would return nothing at all for him. Scanning the whole index is the
        honest thing to do at this size; past a few hundred thousand chunks it
        wants a real filtered index instead.
        """
        # Lazily load on first search if the caller never called load() itself.
        if self.vector_store is None and not self.load():
            return []

        if filter is None:
            return self.vector_store.similarity_search(query, k=k)

        return self.vector_store.similarity_search(
            query, k=k, filter=filter, fetch_k=self.vector_store.index.ntotal
        )

    # --- on-disk layout -------------------------------------------------

    @property
    def index_path(self) -> str:
        return os.path.join(self.persist_directory, f"{self.index_name}.faiss")

    @property
    def docstore_path(self) -> str:
        return os.path.join(self.persist_directory, f"{self.index_name}.pkl")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.persist_directory, f"{self.index_name}{MANIFEST_SUFFIX}")

    def _digests(self) -> dict:
        return {
            os.path.basename(path): _sha256(path)
            for path in (self.index_path, self.docstore_path)
        }

    def _write_manifest(self):
        manifest = {
            "index_name": self.index_name,
            "embedding_model": getattr(self.embeddings, "model_name", None),
            "files": self._digests(),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _verify_manifest(self):
        """Raise IndexTrustError unless the on-disk files match the manifest."""
        if not os.path.exists(self.manifest_path):
            raise IndexTrustError(
                f"No manifest at {self.manifest_path}, so this index cannot be "
                f"shown to be the one built here. Loading it would unpickle "
                f"untrusted data. Rebuild with `python -m src.main`, or set "
                f"{TRUST_ENV_VAR}=1 to load it anyway."
            )

        try:
            with open(self.manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            expected = manifest["files"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise IndexTrustError(
                f"Manifest at {self.manifest_path} is unreadable ({exc}). "
                f"Rebuild the index, or set {TRUST_ENV_VAR}=1 to bypass."
            ) from exc

        actual = self._digests()
        mismatched = sorted(
            name for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        if mismatched:
            raise IndexTrustError(
                f"Index files do not match the manifest: {', '.join(mismatched)}. "
                f"The index was modified or replaced since it was built. "
                f"Rebuild it, or set {TRUST_ENV_VAR}=1 to load it anyway."
            )

    # --- persistence ----------------------------------------------------

    def save(self):
        """
        Persists the vector store to disk, plus a manifest of file digests that
        load() checks before it unpickles anything.
        """
        if self.vector_store:
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)
            self.vector_store.save_local(self.persist_directory, index_name=self.index_name)
            self._write_manifest()

    def load(self) -> bool:
        """
        Loads the vector store from disk. Returns True if successful, False if
        there is no index to load.

        Raises IndexTrustError when an index is present but does not match its
        manifest -- a missing index and an unverifiable one are different
        situations and must not collapse into the same falsy return.
        """
        if not os.path.exists(self.index_path):
            return False

        if _trust_override_enabled():
            print(
                f"WARNING: {TRUST_ENV_VAR} is set; loading {self.index_path} "
                f"without verifying it. This unpickles whatever is on disk."
            )
        else:
            self._verify_manifest()

        self.vector_store = FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            index_name=self.index_name,
            # Reached only after the digests matched, or after an explicit
            # operator override.
            allow_dangerous_deserialization=True,
        )
        return True
