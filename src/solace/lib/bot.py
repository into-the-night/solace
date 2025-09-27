from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from google import genai
from pydantic import BaseModel, Field
from datetime import datetime
# from .redis import Redis
from logging import getLogger

logger = getLogger('uvicorn')
from ..config.settings import settings

client = genai.Client(api_key=settings.gemini_api_key)
deepgram_client = DeepgramClient(api_key=settings.deepgram_api_key)
# redis_client = Redis()

class AlertSchema(BaseModel):
    type: str = Field(..., description="The type of problem, mental or physical, or None.")
    message: str = Field(..., description="The message to be sent to the medical professionals")
    severity: str = Field(..., description="The severity of the problem, low, medium, high, or None.")

def speech_to_text(audio_bytes: bytes, language: str = "en") -> str:
    """convert speech to text using deepgram STT."""
    payload: FileSource = {
        "buffer": audio_bytes,
    }
    if language in ["hindi", "hi"]:
        language = "hi"
    elif language in ["english", "en"]:
        language = "en"
    else:
        return "Sorry I don't understand your language."
    response = deepgram_client.listen.rest.v("1").transcribe_file(
        source=payload,
        options=PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language=language,
        )
    )
    data = response.results.channels[0].alternatives[0].transcript
    logger.info(f"Speech to text: {data}")
    return data


async def synthesize_alert(analysis_result: dict) -> AlertSchema:
    """synthesize alert using Gemini"""
    try:
        facial_emotions = []
        logger.error(f"Analysis result: {analysis_result}")
        if "frame_analysis" in analysis_result:
            for frame in analysis_result["frame_analysis"]:
                if "result" in frame:
                    if "predictions" in frame["result"][0]:
                        if "predictions" in frame["result"][0]["predictions"]:
                            pred = frame["result"][0]["predictions"]["predictions"]
                            if pred:
                                conf = pred[0]["confidence"]
                                pred_class = pred[0]["class"]
                                facial_emotions.append({
                                    "confidence": conf,
                                    "class": pred_class
                                })

        if "audio_analysis" in analysis_result:
            voice_sentiment = f"The person is feeling {analysis_result['audio_analysis']['predicted_emotion']} with a confidence of {analysis_result['audio_analysis']['probabilities'][analysis_result['audio_analysis']['predicted_emotion']]}"
        else:
            voice_sentiment = ""
    
    except:
        facial_emotions = analysis_result["frame_analysis"]
        voice_sentiment = analysis_result["audio_analysis"]
    
    prompt = f"""
    You are a mental healthcare assistant named Maitri, you will talk to the user and adapt according to their tone and mental health condition and always answer in a friendly tone.
    
    Here is an analysis of the astronaut's blog featuring their facial emotions and voice sentiment.
    Facial Emotions: {facial_emotions}
    Voice Sentiment: {voice_sentiment}
    
    Please judge the situation and generate a suitable Alert for the ground zero agency and medical professionals. Focus on negative emotions and be concise.
    
    You MUST respond with a JSON object containing these fields:
    {{
        "type": The type of problem (mental, physical, or none)
        "message": A concise message for medical professionals
        "severity": The severity level (low, medium, high, or none)
    }}
    """
    logger.error(f"Prompt: {prompt}")
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-pro",
            contents=[prompt],
        )
        logger.error(f"Response: {response}")
        if not response.text:
            return AlertSchema(
                type="technical",
                message="Failed to generate alert - empty response from Gemini",
                severity="high"
            )
            
        # Try to parse the response as JSON
        try:
            import json
            response_text = response.text
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            alert_data = json.loads(response_text)
            return AlertSchema(**alert_data)
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, return the raw text as a message
            return AlertSchema(
                type="technical",
                message=f"Failed to parse Gemini response as JSON: {response.text[:200]}",
                severity="high"
            )
            
    except Exception as e:
        return AlertSchema(
            type="technical",
            message=f"Failed to generate alert: {str(e)}",
            severity="high"
        )


