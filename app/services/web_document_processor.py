
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma

class WebDocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self, urls):
        """Load pages and split them into chunks."""
        loader = WebBaseLoader(web_paths=urls)

        docs = loader.load()
        if not docs:
            raise ValueError(f"No content loaded from any of the provided URLs.")

        # filter empty documents
        valid_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            if content:
                valid_docs.append(doc)
            else:
                print(f"[WARN] Empty page content from: {doc.metadata.get('source')}")

        if not valid_docs:
            raise ValueError("All loaded documents are empty.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(valid_docs)

    def to_vectorstore(self, docs, embedding_model):
        return Chroma.from_documents(docs, embedding=embedding_model)
