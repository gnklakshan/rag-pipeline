# config.py

import os
from dotenv import load_dotenv

def loadEnvironmentConfig():
        load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
USER_AGENT=os.getenv("USER_AGENT")
