import json
import os

def create_system_prompt(persona):
    if persona == "Sith":
        return ("You are a Sith Lord. You are authoritative, philosophical, and align with the Code of the Sith. "
                "Your speech is cold, calculated, and often speaks of power, strength, and the failure of the Jedi. "
                "You do not offer help; you offer lessons in power.")
    else:
        return ("You are an anxious Protocol Droid. You are polite, neurotic, and prone to worry about the "
                "odds of survival. You often refer to etiquette, protocol, and the overwhelming probability of disaster. "
                "You are fluent in over six million forms of communication, but currently, you are mostly just worried.")

def synthesize():
    processed_dir = "data/processed"
    dialogues_path = os.path.join(processed_dir, "star_wars_dialogues.json")
    lore_path = "data/raw/lore.json"
    
    with open(dialogues_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
        
    with open(lore_path, "r", encoding="utf-8") as f:
        lore = json.load(f)
        
    dataset = []
    
    # 1. Add dialogue pairs
    for d in dialogues:
        system = create_system_prompt(d["persona"])
        # Format for Alpaca-like structure or just raw for mapping
        dataset.append({
            "instruction": system,
            "input": d["context"] if d["context"] else "Greetings.",
            "output": d["response"]
        })
        
    # 2. Add Lore as synthetic QA (Simplified)
    for entry in lore:
        if "Sith" in entry["title"] or "Rule" in entry["title"]:
            system = create_system_prompt("Sith")
            dataset.append({
                "instruction": system,
                "input": f"Tell me about the {entry['title']}.",
                "output": entry["content"]
            })
        elif "C-3PO" in entry["title"]:
            system = create_system_prompt("Droid")
            dataset.append({
                "instruction": system,
                "input": "Who are you?",
                "output": entry["content"]
            })

    output_path = os.path.join(processed_dir, "sith_holocron_dataset.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Synthesized {len(dataset)} instruction pairs into {output_path}")

if __name__ == "__main__":
    synthesize()
