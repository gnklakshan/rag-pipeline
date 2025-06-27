import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.services.getRelevantWebs import WebSearchService
from app.services.web_document_processor import WebDocumentProcessor

def rag_pipeline(query: str, system_prompt: str, reference_urls: list[str]):
    # Ensure Gemini API key
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise EnvironmentError("GOOGLE_API_KEY must be set for Gemini chat and embeddings")

    # Initialize Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        timeout=None,
        max_retries=2
    )

    # Initialize Gemini embeddings
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # If user didn't supply URLs, fallback to DuckDuckGo search
    if not reference_urls:
        searcher = WebSearchService()
        reference_urls = searcher.search(query)
        if not reference_urls:
            raise ValueError("No URLs found via search; please provide reference URLs.")

    # Load web documents and split into chunks
    processor = WebDocumentProcessor()
    docs = processor.load_and_split(reference_urls)
    if not docs:
        raise ValueError("Failed to load any content from the provided URLs.")

    # Build vector store using Gemini embeddings
    vectordb = processor.to_vectorstore(docs, embedder)
    retriever = vectordb.as_retriever()

    # Build the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    result = qa.invoke({"query": query})

    return result
