
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent.agent import root_agent


# Application configuration
APP_NAME = "banking_operations_copilot"

app = FastAPI(
    title="Enterprise Banking Operations Copilot",
    description="API for the banking operations AI agent",
    version="1.0.0"
)


# ADK session management
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)


# Request schema
class ChatRequest(BaseModel):
    user_id: str
    question: str
    session_id: str | None = None


@app.get("/")
async def root():
    return {"message": "Banking Operations Copilot API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(request: ChatRequest):

    session_id = request.session_id or str(uuid.uuid4())

    # Create session if it doesn't exist
    existing_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=request.user_id,
        session_id=session_id
    )

    if existing_session is None:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=request.user_id,
            session_id=session_id
        )

    # Convert user question into an ADK message
    message = types.Content(
        role="user",
        parts=[types.Part(text=request.question)]
    )

    # Execute ADK agent
    final_response = ""

    async for event in runner.run_async(
        user_id=request.user_id,
        session_id=session_id,
        new_message=message
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    final_response += part.text

    return {
        "user_id": request.user_id,
        "session_id": session_id,
        "question": request.question,
        "answer": final_response
    }
