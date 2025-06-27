import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# web_document_processor.py

class loadWebAndSplit:
    def loadWebContents(self,weblinks:list):
        web_loader = WebBaseLoader(
            web_paths=weblinks,
            bs_kwargs = dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
        )

        docs = web_loader.load()

        return docs

    def splitWebContent(self,chunckSize,chunckOverlap,docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunckSize,chunk_overlap=chunckOverlap)
        return splitter.split_documents(docs)

    def saveToVectorStore(self,documents,embedingModel):
        return Chroma.from_documents(
            documents=documents,
            embedding=embedingModel
        )


