# 데이터 불러오기 -> 청킹 -> 벡터화

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"


#1. 데이터

textPath = "./_data/rag_data/"
loader1 = TextLoader(file_path = textPath + "samsung_outlook.txt", encoding = 'utf-8')
loader2 = TextLoader(file_path = textPath + "nvidia_outlook.txt", encoding = 'utf-8')

text_splitter= RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,  # chunk_overlap은 chunk_size보다 작아야함
    separators = ["\n\n", "\n", " ", ""]
)


split_doc1 = loader1.load_and_split(text_splitter = text_splitter)
split_doc2 = loader2.load_and_split(text_splitter = text_splitter)

# 문서 개수 확인
print(split_doc1)
print(len(split_doc1), len(split_doc2))  # 9 9


from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",   # 임베딩 벡터의 차원 : 1536
    # model = "text-embedding-3-large",     # 임베딩 벡터의 차원 : 3072
    api_key = api_key,
    base_url= base_url,
    # dimensions=5,  # 임베딩 벡터의 dimension 조절 가능
)

DB_PATH = './_db/Chroma11/'

# 저장

db = Chroma.from_documents(
    documents= split_doc1 + split_doc2,
    embedding = embeddings,
    persist_directory = DB_PATH,
    collection_name = 'chroma11',

)


print("Chroma 문서저장 끝")

# 계속 실행할 경우 chroma.sqlite3 파일에 계속 누적됨