from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader, DirectoryLoader


## Only one file read:
txt_one_loader = TextLoader("data/love_story.txt", encoding="utf-8")
txt_doc = txt_one_loader.load()
#print(txt_doc)

pdf_one_loader = PyMuPDFLoader("data/Nestle_HR_Policy.pdf")
pdf_doc = pdf_one_loader.load()
#print(pdf_doc)


##Complete directory Read:
txt_loader = DirectoryLoader(
    path="data",
    glob = "**/*.txt",
    loader_cls= TextLoader,
    loader_kwargs = {'encoding' : 'utf-8'},
    show_progress=True
    )

pdf_loader = DirectoryLoader(
    path="data",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

txt_documents = txt_loader.load()
pdf_documents = pdf_loader.load()

all_docs = txt_documents + pdf_documents
print(all_docs)
