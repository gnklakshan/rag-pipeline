# app/config.py
import os
from dotenv import load_dotenv

def load_environment_config():
    load_dotenv()
    required = ["GOOGLE_API_KEY", "DUCKDUCKGO_API", "GOOGLE_API_KEY","USER_AGENT"]
    for key in required:
        if not os.getenv(key):
            raise EnvironmentError(f"{key} must be set in your .env file")
