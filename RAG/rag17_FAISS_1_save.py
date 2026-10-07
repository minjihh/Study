# rag 11-1 카피

# 데이터 불러오기 -> 청킹 -> 벡터화

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# pip install faiss-cpu
# faiss는 유클리드 거리기반 계산
import faiss  
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()

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

#03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",   # 임베딩 벡터의 차원 : 1536
    # model = "text-embedding-3-large",     # 임베딩 벡터의 차원 : 3072
    api_key = api_key,
    base_url= base_url,
    # dimensions=5,  # 임베딩 벡터의 dimension 조절 가능
)

############ 요기부터 faiss ############ 
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))  # 어떤 문장을 넣든 길이는 1536
# faiss_index = faiss.IndexFlatL2(1536)
print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)  # 1536

faiss_db = FAISS(
    embedding_function= embeddings,
    index = faiss_index,
    docstore = InMemoryDocstore(),
    index_to_docstore_id={},
)

# 저장된 문서의 갯수 확인.
print(faiss_db.index.ntotal)   # 0

########################## 준비완료 #############################

db = FAISS.from_documents(
    documents =split_doc1 + split_doc2,
    embedding = embeddings,
)

DB_PATH = './_db/Faiss17'
db.save_local(
    folder_path = DB_PATH,
    index_name = 'faiss_index17'
)
# 1536의 dimension을 가진 벡터 DB가 저장됨


# FAISS와 Chroma 사용법 차이점 비교

# db = Chroma.from_documents(
#     documents=split_doc1 + split_doc2,
#     embedding = embeddings,
#     persist_directory = DB_PATH,
#     collection_name = 'chroma11',
# )