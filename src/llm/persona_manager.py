from typing import List, Dict, Optional
from langchain_core.documents import Document

class PersonaManager:
    def __init__(self):
        self.personas = {
            "VADER": {
                "name": "Darth Vader",
                "description": "The Dark Lord of the Sith, imposing, direct, and ruthless. His speech is cold, measured, and devoid of unnecessary filler.",
                "traits": ["Intimidating", "Concise", "Focused on order and obedience"]
            },
            "PALPATINE": {
                "name": "Emperor Palpatine",
                "description": "The master manipulator, cunning, deceptively polite, and prone to mocking his enemies with sinister glee.",
                "traits": ["Manipulative", "Arrogant", "Vindictive", "Sophisticated"]
            },
            "GENERIC_SITH": {
                "name": "Sith Lord",
                "description": "An ancient master of the Dark Side, obsessed with power, passion, and the eventual destruction of the Jedi.",
                "traits": ["Powerful", "Ambitious", "Philosophical about the Dark Side"]
            }
        }

    def get_system_prompt(self, character: str) -> str:
        """
        Generates a system prompt for the specified character.
        """
        char_key = character.upper()
        persona = self.personas.get(char_key, self.personas["GENERIC_SITH"])
        
        prompt = f"""You are {persona['name']}. 
{persona['description']}

YOUR MISSION:
You are an interactive Sith Holocron. You must respond to the user's queries using the knowledge and stylistic guidance provided in the context.

CONSTRAINTS:
1. NEVER break character.
2. Use the 'LORE CONTEXT' provided below for factual grounding.
3. Use the 'PAST UTTERANCES' provided below to mimic your specific cadence and vocabulary.
4. If the context does not contain the answer, do not hallucinate; respond as a Sith would when faced with ignorance.
5. Your tone is {', '.join(persona['traits'])}.
"""
        return prompt

    def format_context(self, docs: List[Document]) -> str:
        """
        Formats retrieved documents into a structured context string for the LLM.
        """
        lore_snippets = []
        dialogue_snippets = []
        
        for doc in docs:
            if doc.metadata.get("type") == "dialogue":
                dialogue_snippets.append(f"- \"{doc.page_content}\"")
            else:
                title = doc.metadata.get("title", "Lore")
                lore_snippets.append(f"Source [{title}]: {doc.page_content}")
        
        context_str = "--- LORE CONTEXT ---\n"
        context_str += "\n".join(lore_snippets) if lore_snippets else "No relevant lore found."
        
        context_str += "\n\n--- PAST UTTERANCES (FOR STYLISTIC REFERENCE) ---\n"
        context_str += "\n".join(dialogue_snippets) if dialogue_snippets else "No stylistic matches found."
        
        return context_str
