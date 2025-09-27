from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from ..config.settings import settings

deepgram_client = DeepgramClient(api_key=settings.deepgram_api_key)

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
    return data