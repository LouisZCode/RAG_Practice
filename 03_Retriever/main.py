from plistlib import load
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

load_dotenv()

loader = PyMuPDFLoader(
    "data/Nestle_HR_Policy.pdf"
)

policy = loader.load()
#print(policy)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(policy)

""" for i, chunk in enumerate(chunks):
    print(chunk) """

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("HR_Policy_Vector")