from fastapi import APIRouter

from ..models.requests import ChatRequest
from ..models.responses import ChatResponse
from ...lib.redis import Redis
from ...lib.bot import invoke_llm

router = APIRouter(prefix="/chat")

redis_client = Redis()


@router.post("/message", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Add user message to history
    user_message = {
        "role": "user",
        "content": request.message
    }
    redis_client.add_message(request.user_id, user_message)
    history = redis_client.get_recent_messages(request.user_id, limit=10)
    
    response = await invoke_llm(history)
    
    # Add assistant response to history
    assistant_message = {
        "role": "assistant",
        "content": response
    }
    redis_client.add_message(request.user_id, assistant_message)
    
    return ChatResponse(
        message=response,
    )