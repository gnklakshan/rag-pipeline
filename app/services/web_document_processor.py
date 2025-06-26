import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class loadWebAndSplit:
    def loadWebContents(self,weblinks):
        web_loader = WebBaseLoader(
            web_paths=[weblinks],
            bs_kwargs = dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
        )

        docs = web_loader.load()
        return docs

    def splitWebContent(self,chunckSize,chunckOverlap,docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunckSize,chunk_overlap=chunckOverlap)
        return splitter.split_documents(docs)



