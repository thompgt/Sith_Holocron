import os
import time
from collections.abc import Generator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.observability import metrics

load_dotenv()

class GeminiChatWrapper:
    def __init__(
        self,
        api_key: str | None = None,
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
    ) -> list[BaseMessage]:
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
        metrics.LLM_PROMPT_CHARACTERS.labels(
            model=self.model_name, method="chat"
        ).inc(sum(len(str(m.content)) for m in messages))

        started = time.perf_counter()
        try:
            response = self.llm.invoke(messages)
        except Exception:
            metrics.LLM_REQUESTS.labels(
                model=self.model_name, method="chat", outcome="error"
            ).inc()
            metrics.LLM_STREAM_DURATION.labels(
                model=self.model_name, outcome="error"
            ).observe(time.perf_counter() - started)
            raise

        metrics.LLM_REQUESTS.labels(
            model=self.model_name, method="chat", outcome="success"
        ).inc()
        metrics.LLM_STREAM_DURATION.labels(
            model=self.model_name, outcome="success"
        ).observe(time.perf_counter() - started)
        metrics.LLM_RESPONSE_CHARACTERS.labels(model=self.model_name).inc(
            len(response.content or "")
        )
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
        metrics.LLM_PROMPT_CHARACTERS.labels(
            model=self.model_name, method="stream_chat"
        ).inc(sum(len(str(m.content)) for m in messages))

        started = time.perf_counter()
        first_chunk_seen = False
        try:
            for chunk in self.llm.stream(messages):
                if not first_chunk_seen:
                    first_chunk_seen = True
                    # Model think time only. The endpoint-level TTFT histogram
                    # additionally covers retrieval and prompt assembly; the gap
                    # between the two is where the RAG layer's cost shows up.
                    metrics.LLM_TIME_TO_FIRST_CHUNK.labels(
                        model=self.model_name
                    ).observe(time.perf_counter() - started)
                metrics.LLM_STREAM_CHUNKS.labels(model=self.model_name).inc()
                metrics.LLM_RESPONSE_CHARACTERS.labels(model=self.model_name).inc(
                    len(chunk.content or "")
                )
                yield chunk.content
        except Exception:
            # Errors here propagate to the API layer, where the SSE generator
            # swallows them into a 200-status error frame. Recording the outcome
            # at this level too means the LLM failure is attributable even when
            # the caller is the CLI, which has no HTTP metrics at all.
            metrics.LLM_REQUESTS.labels(
                model=self.model_name, method="stream_chat", outcome="error"
            ).inc()
            metrics.LLM_STREAM_DURATION.labels(
                model=self.model_name, outcome="error"
            ).observe(time.perf_counter() - started)
            raise

        metrics.LLM_REQUESTS.labels(
            model=self.model_name, method="stream_chat", outcome="success"
        ).inc()
        metrics.LLM_STREAM_DURATION.labels(
            model=self.model_name, outcome="success"
        ).observe(time.perf_counter() - started)
