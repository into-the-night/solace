import asyncio
import io
import wave
from google import genai
from google.genai import types
from fastapi import WebSocket, WebSocketDisconnect, APIRouter

from ...config.settings import settings

router = APIRouter(prefix="/chat")
client = genai.Client(api_key=settings.gemini_api_key)

MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful mental healthcare assistant named Maitri, you will talk to the user and adapt according to their tone and mental health condition and always answer in a friendly tone.",
}


@router.websocket("/audio")
async def live_audio_ws(websocket: WebSocket):
    """
    Bi-directional live audio chat:
    - Receive raw PCM16 audio from client
    - Forward to Gemini Live API
    - Stream Gemini audio output back to client
    """
    await websocket.accept()

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            
            async def receive_from_client():
                """Forward client audio to Gemini session"""
                try:
                    while True:
                        msg = await websocket.receive_bytes()
                        await session.send_realtime_input(
                            audio=types.Blob(data=msg, mime_type="audio/pcm;rate=16000")
                        )
                except WebSocketDisconnect:
                    pass

            async def send_to_client():
                """Stream audio responses from Gemini to client"""
                async for response in session.receive():
                    if response.data:
                        await websocket.send_bytes(response.data)

            # Run both directions concurrently
            await asyncio.gather(receive_from_client(), send_to_client())

    except WebSocketDisconnect:
        print("Client disconnected")
