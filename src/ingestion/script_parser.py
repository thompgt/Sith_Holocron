
import pandas as pd
from langchain_core.documents import Document


class ScriptParser:
    def __init__(self):
        pass

    def parse_csv(self, file_path: str, sep: str = ";", char_col: str = "from", text_col: str = "text") -> list[Document]:
        """
        Parses a CSV script file.
        """
        df = pd.read_csv(file_path, sep=sep)
        docs = []
        for _, row in df.iterrows():
            character = str(row[char_col]).strip()
            text = str(row[text_col]).strip()

            if not text or text == "nan":
                continue

            metadata = {
                "character": character,
                "source": file_path,
                "type": "dialogue"
            }
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def parse_tab_txt(self, file_path: str) -> list[Document]:
        """
        Parses a tab-separated TXT script file (like EpisodeIV_dialogues.txt).
        """
        docs = []
        with open(file_path, encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or "\t" not in line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                character = parts[0].strip()
                text = parts[1].strip()

                metadata = {
                    "character": character,
                    "source": file_path,
                    "type": "dialogue"
                }
                docs.append(Document(page_content=text, metadata=metadata))
        return docs
