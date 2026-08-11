"""Heuristic evaluation of a full RAG cycle.

This lived at tests/persona_audit.py, where it was neither a test nor reachable:
pytest only collects files matching test_*.py, so the module was never imported
by anything and its one bug sat undetected. It is library code -- a harness you
call with a live Gemini wrapper -- so it belongs under src/. The unit tests for
it are in tests/test_persona_auditor.py.
"""

class PersonaAuditor:
    """Runs a query through retrieval + generation and scores the result.

    The checks are deliberately crude. Real evaluation would need a grader model
    and a rubric; these catch the failures that are obvious from the text alone,
    which is the class of failure that actually shipped here (an in-character
    voice that ignored its retrieved context, or a response with no persona
    signal in it at all).
    """

    #: A response shorter than this is treated as a non-answer -- a refusal, an
    #: error string, or a truncated stream -- rather than something to grade.
    MIN_SUBSTANTIVE_CHARS = 40

    def __init__(self, gemini_wrapper, retriever, persona_manager):
        self.llm = gemini_wrapper
        self.retriever = retriever
        self.pm = persona_manager

    def _character_check(self, character: str, response: str) -> bool:
        """Does the response name the persona or any of its known aliases?

        The old form was:

            character.upper() in response.upper()
            or self.pm.personas.get(character.upper())["name"].upper() in ...

        which subscripts the result of .get(), so any character not in the
        persona table -- exactly the input an audit is most likely to be fed --
        raised TypeError: 'NoneType' object is not subscriptable instead of
        reporting a failed check. PersonaManager.aliases() already resolves
        unknown characters to GENERIC_SITH, so routing through it removes the
        None case entirely and picks up alias spellings (SIDIOUS for PALPATINE)
        that the two-way comparison missed.
        """
        haystack = response.upper()
        if character and character.upper() in haystack:
            return True

        persona = self.pm.personas.get(character.upper()) if character else None
        if persona and persona["name"].upper() in haystack:
            return True

        aliases = self.pm.aliases(character) or set()
        return any(alias in haystack for alias in aliases)

    @staticmethod
    def _context_check(docs, response: str) -> bool:
        """Did any retrieved lore passage visibly survive into the response?

        A prefix match is a weak proxy for grounding -- a paraphrase scores zero
        -- so read a False as "not proven grounded", not as "hallucinated".
        """
        haystack = response.lower()
        return any(
            doc.page_content[:20].lower() in haystack
            # Default the key rather than requiring it, matching HybridRetriever.
            # LoreProcessor now stamps type="lore", but an index built before
            # that change has chunks with no type and must still count as lore.
            for doc in docs
            if doc.metadata.get("type", "lore") != "dialogue" and doc.page_content
        )

    def audit_response(self, character: str, query: str, k: int = 4) -> dict:
        docs = self.retriever.retrieve(query, character=character, k=k)
        context = self.pm.format_context(docs)
        system_prompt = self.pm.get_system_prompt(character)
        response = self.llm.chat(system_prompt, query, context)

        return {
            "response": response,
            "character_check": self._character_check(character, response),
            "context_used": self._context_check(docs, response),
            "retrieved": len(docs),
            "length": len(response),
            "substantive": len(response) >= self.MIN_SUBSTANTIVE_CHARS,
        }

    def audit_all(self, queries: dict, k: int = 4) -> dict:
        """Audit a {character: [query, ...]} mapping in one pass."""
        return {
            character: [self.audit_response(character, q, k=k) for q in qs]
            for character, qs in queries.items()
        }


def summarize(results: dict) -> dict:
    """Collapse audit_all output into pass rates."""
    flat = [r for rs in results.values() for r in rs]
    total = len(flat)
    if not total:
        return {"total": 0}
    return {
        "total": total,
        "in_character": sum(r["character_check"] for r in flat) / total,
        "grounded": sum(r["context_used"] for r in flat) / total,
        "substantive": sum(r["substantive"] for r in flat) / total,
    }
