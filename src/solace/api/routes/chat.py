import asyncio
import io
import wave
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fastapi import WebSocket, WebSocketDisconnect, APIRouter

from ...config.settings import settings
from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ...lib.redis import Redis

router = APIRouter(prefix="/chat")
client = genai.Client(api_key=settings.gemini_api_key)

redis_client = Redis()

class ResponseFormat(BaseModel):
    response: str = Field(..., description="The response to the user's latest message")
    sentiment_score: int = Field(..., description="The sentiment score of the response 0 being the most negative and 100 being the most positive")


async def invoke_gemini(history: list):
    """
    Chat with a bot and provide a response to the farmer in a friendly and easy to understand manner.
    """
    conversation_context = ""
    if history:
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            conversation_context += f"{role.capitalize()}: {content}\n\n"
    
    prompt = f"""
        You are a mental healthcare assistant named Maitri, you will talk to the user and adapt according to their tone and mental health condition and always answer in a friendly tone.
        
        Here is the conversation history:
        {conversation_context}
        
        Please provide a helpful response to the user's latest message in the following format taking into account the conversation context.
        """
    
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": ResponseFormat,
        },
    )
    response_text = response.parsed.response
    sentiment_score = response.parsed.sentiment_score
    return response_text, sentiment_score


@router.post("/message", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Add user message to history
    user_message = {
        "role": "user",
        "content": request.message
    }
    redis_client.add_message(request.user_id, user_message)
    history = redis_client.get_recent_messages(request.user_id, limit=10)
    
    response = await invoke_gemini(history)
    
    # Add assistant response to history
    assistant_message = {
        "role": "assistant",
        "content": response
    }
    redis_client.add_message(request.user_id, assistant_message)
    
    return ChatResponse(
        message=response,
    )