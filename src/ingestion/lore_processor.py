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

            # Extract metadata.
            #
            # "type" was absent here while ScriptParser set type="dialogue", so
            # real lore chunks carried no type at all. HybridRetriever survived
            # that by defaulting the key to "lore", but PersonaAuditor's
            # grounding check compared == "lore" and therefore scored every
            # response against a real index as ungrounded. Every test that
            # touches lore builds its Documents by hand with type="lore", so
            # nothing caught the mismatch. Set it at the source instead of
            # teaching each consumer to default it.
            metadata = {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "source": file_path,
                "type": "lore",
            }

            # Split content into chunks
            chunks = self.text_splitter.split_text(content)

            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata=metadata))

        return all_docs
