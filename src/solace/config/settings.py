from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    roboflow_api_url: str = "https://serverless.roboflow.com"
    roboflow_api_key: str = os.getenv("ROBOFLOW_API_KEY")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY")

settings = Settings()