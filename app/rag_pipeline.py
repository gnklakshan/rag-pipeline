# rag_pipeline.py

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import os
from app.config import loadEnvironmentConfig
from app.services.getRelevantWebs import webContent
from app.services.web_document_processor import loadWebAndSplit


def rag_pipeline( searchInput, system_prompt, userInput):
    loadEnvironmentConfig()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


    # gemini caht model to get respond from generative ai
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    # convert_system_message_to_human =True,
    timeout=None,
    max_retries=2 )


    # demini embedding model to get vector embedding
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
        )

    # create intance
    web = webContent()
    webContentProcessor = loadWebAndSplit()

    # get web results
    webSearchRespons = web.getWebContents(searchInput)

    # here we only consider first 20 web results to reduce cost
    webLinks = [response["link"] for response in web.toJsonObject(webSearchRespons)][:20]

    print(f"get web links -{len(webLinks)} no of links received")
    print(webLinks)

    # load web results
    docs = webContentProcessor.loadWebContents(webLinks)
    print(f"docs == {docs}")

    non_empty_docs = [doc for doc in docs if doc.page_content.strip()]

    if not non_empty_docs:
        raise ValueError("No document content found in the provided URLs.")

    # split the content
    splits = webContentProcessor.splitWebContent(chunckSize=1000,chunckOverlap=200,docs=non_empty_docs)

    print(f"Total splits: {len(splits)}")  #for log purpose

    if not splits:
        raise ValueError("No document splits found — check web content loading or parsing.")

    # convert to vector
    vectorStore = webContentProcessor.saveToVectorStore(documents=splits,embedingModel=gemini_embeddings)

    retriever = vectorStore.as_retriever()

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answering_chain)

    return rag_chain.invoke({"input":userInput})
