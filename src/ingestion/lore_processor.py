import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LoreProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_file(self, file_path: str) -> list[Document]:
        """
        Loads a JSON lore file and returns a list of chunked Documents.
        """
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)

        all_docs = []
        for item in data:
            content = item.get("content", "")
            if not content:
                continue

            # Extract metadata
            metadata = {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "source": file_path
            }

            # Split content into chunks
            chunks = self.text_splitter.split_text(content)

            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata=metadata))

        return all_docs
