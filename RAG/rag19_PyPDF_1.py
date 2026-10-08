from langchain_community.document_loaders import TextLoader, PyPDFLoader

path = './_data/'
pdf_loader = PyPDFLoader(path + "Attention is all you need.pdf")
pdf_docs = pdf_loader.load()

print(type(pdf_docs))  # <class 'list'>
print(len(pdf_docs))   # 15 -> 페이지 수
print(pdf_docs)   # Document안에 metadata와 page_content가 있다. 

print("=========================================")
print(pdf_docs[0])
print("=========================================")

