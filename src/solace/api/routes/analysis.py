import cv2
import torch
import librosa
import tempfile, soundfile as sf
from inference_sdk import InferenceHTTPClient
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from transformers import AutoProcessor, AutoModelForAudioClassification
import os
import uuid
import moviepy.editor as mp

from ...config.settings import settings
from ..models.responses import AnalysisResponse
from ...lib.bot import speech_to_text

router = APIRouter(prefix="/analysis")

# Load model once at startup
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

from transformers import Wav2Vec2FeatureExtractor
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Only quantize Linear layers
    dtype=torch.float16
)

client = InferenceHTTPClient(
    api_url=settings.roboflow_api_url,
    api_key=settings.roboflow_api_key
)


def predict_emotion(audio_bytes: bytes) -> dict:
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Load audio at 16kHz
    speech, sr = librosa.load(tmp_path, sr=16000)

    # Preprocess
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    # Inference
    with torch.no_grad():
        logits = quantized_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Get prediction + probabilities
    predicted_class_id = torch.argmax(probs).item()
    predicted_label = quantized_model.config.id2label[predicted_class_id]
    print(predicted_label)
    return {
        "predicted_emotion": predicted_label,
        "probabilities": {
            quantized_model.config.id2label[i]: float(p)
            for i, p in enumerate(probs)
        }
    }

@router.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = predict_emotion(audio_bytes)
    return result


def extract_frames(video_path: str, output_dir: str, interval: int = 5):
    """Extract frames from video at specified interval in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)

        frame_count += 1

    cap.release()
    return frames

@router.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    # Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        video_bytes = await file.read()
        tmp_video.write(video_bytes)
        tmp_video_path = tmp_video.name

    # Prepare temp files for output
    tmp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(tmp_dir, "input")
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    audio_output_path = os.path.join(input_dir, f"audio_only_{uuid.uuid4().hex}.wav")

    try:
        # Load video
        try:
            clip = mp.VideoFileClip(tmp_video_path)
        except Exception as e:
            return {"error": f"Failed to load video file: {str(e)}"}

        # Extract audio
        if clip.audio is not None:
            try:
                clip.audio.write_audiofile(audio_output_path, verbose=False, logger=None)
            except Exception as e:
                return {"error": f"Failed to write audio file: {str(e)}"}
        else:
            return {"error": "No audio track found in the uploaded video."}

        # Extract frames every 5 seconds
        try:
            frame_paths = extract_frames(tmp_video_path, frames_dir)
        except Exception as e:
            return {"error": f"Failed to extract frames: {str(e)}"}

        # Process frames through Roboflow workflow
        frame_results = []
        for frame_path in frame_paths:
            result = client.run_workflow(
                workspace_name="sih-n7y20",
                workflow_id="face-it",
                images={
                    "image": frame_path
                },
                use_cache=True
            )
            frame_results.append({
                "frame": os.path.basename(frame_path),
                "result": result
            })

        # Process audio
        with open(audio_output_path, "rb") as f_aud:
            audio_only_bytes = f_aud.read()
            audio_text = speech_to_text(audio_only_bytes)
            audio_result = predict_emotion(audio_only_bytes)

        return {
            "frame_analysis": frame_results,
            "audio_analysis": audio_result,
            "audio_text": audio_text,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up all temporary files and directories
        try:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")


