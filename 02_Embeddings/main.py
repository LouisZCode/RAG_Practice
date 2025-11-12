from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyMuPDFLoader(
    "data/Nestle_HR_Policy.pdf"
)

policy = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(policy)

print(f"We got {len(chunks)} from this splitting")

for i, chunk in enumerate(chunks[:5]):
    print()
    #print(f"This is the metadaata: {chunk.metadata}")
    print(f"This is the content:\n{chunk.page_content[:100]}...")
    print(f"And it is {len(chunk.page_content)} characters long")