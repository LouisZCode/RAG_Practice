from langchain_community.document_loaders import PyMuPDFLoader


from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyMuPDFLoader("data/Nestle_HR_Policy.pdf")
policy = loader.load()
#print(policy)

#Text split and get the Chunks:

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(policy)


# Quick and effective chunk view
for i, chunk in enumerate(chunks[:5]): #first five
    print(len(chunks))
    print(f"\n--- Chunk {i+1} ---")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Preview: {chunk.page_content[:200]}...")
    print(f"Length: {len(chunk.page_content)} chars")

