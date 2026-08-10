import json

import pytest

from src.ingestion.lore_processor import LoreProcessor


@pytest.fixture
def sample_lore_json(tmp_path):
    data = [
        {
            "url": "https://wiki.com/sith",
            "title": "Sith Code",
            "content": "Peace is a lie. There is only Passion."
        },
        {
            "url": "https://wiki.com/jedi",
            "title": "Jedi Code",
            "content": "There is no emotion, there is peace."
        }
    ]
    p = tmp_path / "test_lore.json"
    p.write_text(json.dumps(data))
    return str(p)

def test_lore_processor_chunks(sample_lore_json):
    processor = LoreProcessor(chunk_size=10, chunk_overlap=0)
    chunks = processor.process_file(sample_lore_json)

    assert len(chunks) > 0
    assert "Sith Code" in chunks[0].metadata["title"]
    assert "Peace" in chunks[0].page_content

def test_lore_processor_metadata(sample_lore_json):
    processor = LoreProcessor()
    chunks = processor.process_file(sample_lore_json)

    for chunk in chunks:
        assert "url" in chunk.metadata
        assert "title" in chunk.metadata
        assert "source" in chunk.metadata
