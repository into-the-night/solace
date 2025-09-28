from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
# from google import genai
from ollama import chat, ChatResponse
from pydantic import BaseModel, Field
from logging import getLogger

logger = getLogger('uvicorn')
from ..config.settings import settings

# client = genai.Client(api_key=settings.gemini_api_key)


deepgram_client = DeepgramClient(api_key=settings.deepgram_api_key)

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


async def synthesize_alert(analysis_result: dict) -> AlertSchema | str:
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
    
    Please judge the situation and generate a suitable Alert for the ground zero agency and medical professionals. Focus on the overall mood and emotions and be concise.
    
    You MUST respond with a JSON object containing these fields:
    {{
        "type": The type of problem (mental, physical, or none)
        "message": A concise message for medical professionals
        "severity": The severity level (low, medium, high, or none)
    }}
    """
    logger.error(f"Prompt: {prompt}")
    try:
        # response = await client.aio.models.generate_content(
        #     model="gemini-2.5-pro",
        #     contents=[prompt],
        # )

        response : ChatResponse = chat(
            model="gemma3:1b", 
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        logger.error(f"Response: {response}")
        if not response.message.content:
            return "Error occurred while generating alert"
            
        # Try to parse the response as JSON
        try:
            import json
            response_text = response.message.content
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            alert_data = json.loads(response_text)
            return AlertSchema(**alert_data)
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, return the raw text as a message
            return f"Failed to parse Gemini response as JSON: {response.text[:200]}"
            
    except Exception as e:
        return f"Failed to generate alert: {str(e)}"

async def invoke_llm(history: list):
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
    
    response : ChatResponse = chat(
        model="gemma3:1b",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    response_text = response.message.content
    return response_text

