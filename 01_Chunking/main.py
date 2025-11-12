from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/Nestle_HR_Policy.pdf")
document = loader.load()
print(document)