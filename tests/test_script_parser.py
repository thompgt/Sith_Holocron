
import pytest

from src.ingestion.script_parser import ScriptParser


@pytest.fixture
def sample_csv_script(tmp_path):
    # Format like star_wars_1_data.csv
    content = "id;from;to;text;text to check;where;number\n1;QUI-GON;CAPTAIN;Captain.;;INT. ROOM;1"
    p = tmp_path / "prequel.csv"
    p.write_text(content)
    return str(p)

@pytest.fixture
def sample_txt_script(tmp_path):
    # Format like EpisodeIV_dialogues.txt
    content = "STAR WARS - EPISODE 4\n\nTHREEPIO\tDid you hear that?\nVADER\tWhere are those transmissions?"
    p = tmp_path / "ot.txt"
    p.write_text(content)
    return str(p)

def test_parse_csv_script(sample_csv_script):
    parser = ScriptParser()
    docs = parser.parse_csv(sample_csv_script, sep=";", char_col="from", text_col="text")

    assert len(docs) == 1
    assert docs[0].metadata["character"] == "QUI-GON"
    assert docs[0].page_content == "Captain."

def test_parse_txt_tab_script(sample_txt_script):
    parser = ScriptParser()
    docs = parser.parse_tab_txt(sample_txt_script)

    assert len(docs) == 2
    assert docs[0].metadata["character"] == "THREEPIO"
    assert "Did you hear that?" in docs[0].page_content
    assert docs[1].metadata["character"] == "VADER"
