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

""" #This was used before to create the local vector store, now we need to only load it
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("HR_Policy_Vector") """

vector_store = FAISS.load_local("HR_Policy_Vector", embeddings, allow_dangerous_deserialization=True)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

query = "What does my HR Policy speak about? and what company does it apply to?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['source']}")