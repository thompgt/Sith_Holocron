import os
from typing import List, Optional, Generator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

class GeminiChatWrapper:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature
        )

    def _prepare_messages(
        self, 
        system_prompt: str, 
        user_query: str, 
        context: str
    ) -> List[BaseMessage]:
        """
        Prepares the message list for the Gemini model.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nUSER QUERY: {user_query}")
        ]
        return messages

    def chat(
        self, 
        system_prompt: str, 
        user_query: str, 
        context: str
    ) -> str:
        """
        Sends a synchronous chat request.
        """
        messages = self._prepare_messages(system_prompt, user_query, context)
        response = self.llm.invoke(messages)
        return response.content

    def stream_chat(
        self, 
        system_prompt: str, 
        user_query: str, 
        context: str
    ) -> Generator[str, None, None]:
        """
        Streams the chat response.
        """
        messages = self._prepare_messages(system_prompt, user_query, context)
        for chunk in self.llm.stream(messages):
            yield chunk.content
