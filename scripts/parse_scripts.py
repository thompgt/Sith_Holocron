import csv
import json
import os


def parse_original_trilogy():
    dialogues = []
    text_dir = "data/raw/star-wars-scripts/Text_files"
    files = ["EpisodeIV_dialogues.txt", "EpisodeV_dialogues.txt", "EpisodeVI_dialogues.txt"]

    target_chars = ["THREEPIO", "VADER", "SIDIOUS", "EMPEROR"]

    for filename in files:
        filepath = os.path.join(text_dir, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        prev_line = None
        for line in lines:
            if "\t" not in line:
                continue

            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue

            char = parts[0].strip()
            text = parts[1].strip()

            if char in target_chars:
                persona = "Sith" if char in ["VADER", "SIDIOUS", "EMPEROR"] else "Droid"
                dialogues.append({
                    "persona": persona,
                    "character": char,
                    "context": prev_line["text"] if prev_line else "",
                    "response": text
                })

            prev_line = {"char": char, "text": text}

    return dialogues

def parse_prequels():
    dialogues = []
    csv_dir = "data/raw/prequel-csv"
    files = ["star_wars_1_data.csv", "star_wars_2_data.csv", "star_wars_3_data.csv"]

    # In these CSVs, the names might be different
    target_chars = ["VADER", "SIDIOUS", "PALPATINE", "EMPEROR", "DARTH VADER", "DARTH SIDIOUS", "CHANCELLOR PALPATINE"]
    droid_chars = ["THREEPIO", "C-3PO", "TC-14"]

    for filename in files:
        filepath = os.path.join(csv_dir, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            prev_row = None
            for row in reader:
                char = row["from"].strip()
                text = row["text"].strip()

                if char in target_chars or char in droid_chars:
                    persona = "Sith" if char in target_chars else "Droid"
                    dialogues.append({
                        "persona": persona,
                        "character": char,
                        "context": prev_row["text"] if prev_row else "",
                        "response": text
                    })
                prev_row = row

    return dialogues

def main():
    ot_dialogues = parse_original_trilogy()
    prequel_dialogues = parse_prequels()

    all_dialogues = ot_dialogues + prequel_dialogues

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/star_wars_dialogues.json", "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, indent=4)

    print(f"Extracted {len(all_dialogues)} dialogue pairs.")
    sith_count = len([d for d in all_dialogues if d["persona"] == "Sith"])
    droid_count = len([d for d in all_dialogues if d["persona"] == "Droid"])
    print(f"Sith pairs: {sith_count}")
    print(f"Droid pairs: {droid_count}")

if __name__ == "__main__":
    main()
