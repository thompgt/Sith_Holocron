import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.gemini_wrapper import GeminiChatWrapper


@pytest.fixture
def gemini_wrapper():
    return GeminiChatWrapper(api_key="mock_key")

def test_gemini_wrapper_init(gemini_wrapper):
    assert gemini_wrapper.model_name == "gemini-1.5-flash"
    assert gemini_wrapper.llm is not None

def test_prompt_formatting(gemini_wrapper):
    system_prompt = "You are Vader."
    user_query = "Who are you?"
    context = "Context info."

    messages = gemini_wrapper._prepare_messages(system_prompt, user_query, context)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Context info." in messages[1].content
    assert "Who are you?" in messages[1].content
