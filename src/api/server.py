import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.gemini_wrapper import GeminiChatWrapper
from src.llm.persona_manager import PersonaManager

app = FastAPI()

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
pm = PersonaManager()
vs_manager = VectorStoreManager(persist_directory="data/vector_store")
vs_manager.load()
retriever = HybridRetriever(vs_manager)
llm = GeminiChatWrapper()

class ChatRequest(BaseModel):
    message: str
    persona: str = "VADER"

@app.get("/api/personas")
async def get_personas():
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in pm.personas.items()
    ]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    docs = retriever.retrieve(request.message, character=request.persona, k=4)
    context = pm.format_context(docs)
    system_prompt = pm.get_system_prompt(request.persona)
    
    async def event_generator():
        try:
            for chunk in llm.stream_chat(system_prompt, request.message, context):
                if chunk:
                    # SSE format: data: <content>\n\n
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
