from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS


from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool

load_dotenv()

""" loader = PyMuPDFLoader(
    "data/Nestle_HR_Policy.pdf"
)

policy = loader.load()
#print(policy)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(policy) """

""" for i, chunk in enumerate(chunks):
    print(chunk) """

""" #This was used before to create the local vector store, now we need to only load it

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("HR_Policy_Vector") """

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.load_local("HR_Policy_Vector", embeddings, allow_dangerous_deserialization=True)
# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


#This is new:
retriever_tool = create_retriever_tool(
    retriever,
    name="search_documents",
    description="Search through the document knowledge base to find relevant information."
)


System_Prompt = """
You are a internal model for the company.
You alway use your search_documents tool to find context before answering.
You answer honestly, and if the context does not provide an answer, you let the user know."""


agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=System_Prompt,
    tools=[retriever_tool]
)


query = input("Ask something to the HR Policy:\n")

result = agent.invoke({
    "role" : "user",
    "messages" : query
    })

for i,msg in enumerate(result["messages"]):
    msg.pretty_print()