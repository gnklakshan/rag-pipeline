# main.py
import os
import logging
from app.config import load_environment_config
from app.rag_pipeline import rag_pipeline

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Load environment var
    load_environment_config()

    # Ask the user for reference URLs
    raw_urls = input("Enter one or more reference URLs (comma-separated), or leave blank to search automatically: ").strip()
    reference_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

    # Define your prompts
    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    user_input = input("What would you like to know? ").strip()

    # RAG pipeline
    logging.info("Starting RAG pipeline...")
    try:
        answer = rag_pipeline(
            query=user_input,
            system_prompt=system_prompt,
            reference_urls=reference_urls
        )
        print("\n=== RAG Output ===\n")
        print(answer)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print("Sorry, something went wrong. Please check the logs for details.")

if __name__ == "__main__":
    main()
