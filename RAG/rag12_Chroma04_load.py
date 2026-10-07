# rag 12-3 카피
# 이미 Chroma로 저장됐을 경우는 불러오기만 하면 됨 
# 한 폴더에 텍스트문서가 여러개 있을 경우
# 데이터 불러오기 -> 청킹 -> 벡터화


import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

""" DB에 저장하는 부분 주석처리
from glob import glob

path = "./_data/rag_data/"
# 폴더에서 텍스트 파일 목록 가져오기
txt_files = glob(os.path.join(path, '*.txt'))  # 경로내의 모든 txt파일 리스트 형태로 나타내줌

print(txt_files)  
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt',
#  './_data/rag_data\\samsung_outlook.txt']
# 리스트 -> 이터레이터

#01 데이터
data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding = 'utf-8')
    # data.append(loader)
    data += loader.load()

# print(data)
print("=====================================")
print(len(data))
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)  # [8158, 2049, 1898]

#02 문서를 자른다. (청킹)
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size = 300,   # 여기서는 문자기준
    chunk_overlap = 10,  # chunk_overlap은 chunk_size보다 작아야함, 문단이 하나의 chunk안에 들어가지 못하고 잘리거나 하는부분에서 overlap 적용
    separators = ["\n\n", "\n", " ", ""]
)

texts = text_splitter.split_documents(data)
print("생성된 텍스트 청크수 :", len(texts))  # 52 -> chunk 임베딩 벡터가 52개 생길예정
print("각 청크의 길이 :", list(len(text.page_content) for text in texts))
# 각 청크의 길이 : [259, 282, 282, 128, 276, 214, 158, 249, 262, 291, 268, 182, 286, 295, 182, 162, 283, 286, 257, 235, 214, 258, 207, 286, 220, 198, 271, 57, 272, 122, 9, 269, 299, 284, 289, 209,222, 230, 254, 249, 296, 90, 247, 243, 185, 219, 239, 235, 298, 282, 187, 249]

# print("첫번째 청크의 내용 : ", texts[0])
# # 청크안에 들어있는 내용: page_content(내용), metadata(경로)
# print("첫번째 청크의 내용 : ", texts[0].page_content)  # 259
# print("첫번째 청크의 길이 : ", len(texts[0].page_content))
# print("두번째 청크의 내용 : ", texts[1])
"""

#03 임베딩
# OpenAIEmbeddings : tokenizer로 토큰별 값을 준 후에, transformer를 통과해서 문맥정보를 포함시킨뒤에 합해져 chunk embedding 이 됨
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",   # 임베딩 벡터의 차원 : 1536
    # model = "text-embedding-3-large",     # 임베딩 벡터의 차원 : 3072
    api_key = api_key,
    base_url= base_url,
    # dimensions=5,  # 임베딩 벡터의 dimension 조절 가능
)

# sample_text ="삼성전자의 창업자는 누구인가요?"
# vector = embeddings.embed_query(sample_text)   # method와 함수는 같은듯 다르다.
# # print(vector)
# print(len(vector))  # 1536  (chunk의 개수와 다름에 유의)




# exit()
DB_PATH = './_db/Chroma12/'

# 저장
# vector_store = Chroma.from_documents(
#     documents= texts,
#     embedding = embeddings,
#     persist_directory = DB_PATH,
#     collection_name = 'chroma12',

# )

# 불러올대는 documents, embedding 사용하지 않음 
vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = DB_PATH,
    collection_name = 'chroma12',
)


print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}",) 
# vector store에 저장된 문서 수 : 52

query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query)

print(f"검색 결과의 길이: {len(result)}")
# 검색 결과의 길이: 4    
# # similarity_search에서 k값 default가 4였으므로

############### Retrievers ############### 
############### 검색기 ############### 
retriever = vector_store.as_retriever(search_kwargs={"k":2})

print(retriever)
aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수 : {len(aaa)}")
print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}")
print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content}")



