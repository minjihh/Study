# rag 12-4 카피
# openai 연결해서 content 가져오기
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


# exit()
DB_PATH = './_db/Chroma12/'

# 불러올때는 documents, embedding 사용하지 않음 
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


print("====================================================")
####################### 모델 연결 #########################

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5-nano',
    temperature = 0,
    max_tokens = 1000,  # 너무 적으면 답이 안나오는 경우도 있음, default는 무제한
    api_key = api_key,
    base_url = base_url,
)

response = model.invoke("nvida가 집중하는 부분?")  # model.invoke는 model.predict와 동일
print("model의 답변 : ", response.content)
print("====================================================")

query_with_context = f"""
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
"""

# aaa는 위쪽에서 삼성전자 창업주에 대해 물어본 내용에 대한 retriever.invoke 한내용

response = model.invoke(query_with_context)
print("model의 응답 :", response.content)

