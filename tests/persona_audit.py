import pytest
from src.llm.persona_manager import PersonaManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStoreManager

class PersonaAuditor:
    def __init__(self, gemini_wrapper, retriever, persona_manager):
        self.llm = gemini_wrapper
        self.retriever = retriever
        self.pm = persona_manager

    def audit_response(self, character, query):
        """
        Runs a full RAG cycle and returns the response plus an evaluation.
        """
        # 1. Retrieve
        docs = self.retriever.retrieve(query, character=character, k=4)
        context = self.pm.format_context(docs)
        
        # 2. Generate
        system_prompt = self.pm.get_system_prompt(character)
        response = self.llm.chat(system_prompt, query, context)
        
        # 3. Simple Heuristic Checks
        audit_results = {
            "response": response,
            "character_check": character.upper() in response.upper() or self.pm.personas.get(character.upper())["name"].upper() in response.upper(),
            "context_used": any(doc.page_content[:20].lower() in response.lower() for doc in docs if doc.metadata.get("type") == "lore"),
            "length": len(response)
        }
        
        return audit_results

# Note: Actual model-graded evaluation would require a valid API key and a separate "Grader" prompt.
# For Task 10, we establish the framework and basic heuristics.
