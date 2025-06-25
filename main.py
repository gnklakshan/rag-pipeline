# Install required libraries if not already installed
# pip install flask langchain langchain-google-genai langchain-chroma langchain-community beautifulsoup4

import os
from flask import Flask, request, jsonify
import bs4
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

# === Setup ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyAc0f4fV9vJB4bjhd18zEG4VioqnDCQbaQ"

app = Flask(__name__)

# === Load & preprocess documents once on startup ===
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)

system_prompt = (
    "You are an assistant for question answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(chat_model, chat_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# === Define API route ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' in request."}), 400

    try:
        result = rag_chain.invoke({"input": question})
        return jsonify({"answer": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "RAG API is running!"})

# === Run Flask server ===
if __name__ == "__main__":
    app.run(port=5000)
