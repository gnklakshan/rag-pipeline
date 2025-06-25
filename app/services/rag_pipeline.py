from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv() #load .env

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# gemini caht model to get respond from generative ai
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
# convert_system_message_to_human =True,
timeout=None,
max_retries=2 )


# demini embedding model to get vector embedding
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
    )


response = model.invoke("hi")

print(response.content)