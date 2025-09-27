from pydantic import BaseModel

class AnalysisResponse(BaseModel):
    frame_analysis: list[dict]
    audio_analysis: dict

class ChatResponse(BaseModel):
    message: str
