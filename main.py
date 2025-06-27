# main.py
import os

from app.config import loadEnvironmentConfig
from app.rag_pipeline import rag_pipeline


def main():
    # Load environment variables
    loadEnvironmentConfig()

    # Optional: explicitly set the key if needed by Gemini
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    # Define inputs
    searchInput = "What is nextjs framework?"

    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer the question "
        "If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    userInput = "What is nextjs framework?"

    # Run pipeline
    result = rag_pipeline(searchInput, system_prompt, userInput)

    # Display result
    print("\n=== RAG Output ===\n")
    print(result["answer"] if isinstance(result, dict) and "answer" in result else result)

if __name__ == "__main__":
    main()
