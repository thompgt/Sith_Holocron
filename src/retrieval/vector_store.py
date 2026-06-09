import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

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
        self.vector_store: Optional[FAISS] = None

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to the vector store. Creates a new index if one doesn't exist.
        """
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search in the vector store.
        """
        if self.vector_store is None:
            # Try to load if not initialized
            if not self.load():
                return []
        
        return self.vector_store.similarity_search(query, k=k)

    def save(self):
        """
        Persists the vector store to disk.
        """
        if self.vector_store:
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)
            self.vector_store.save_local(self.persist_directory, index_name=self.index_name)

    def load(self) -> bool:
        """
        Loads the vector store from disk. Returns True if successful.
        """
        index_path = os.path.join(self.persist_directory, f"{self.index_name}.faiss")
        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(
                self.persist_directory, 
                self.embeddings, 
                index_name=self.index_name,
                allow_dangerous_deserialization=True
            )
            return True
        return False
