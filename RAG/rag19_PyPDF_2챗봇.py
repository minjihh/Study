# 챗봇까지 맹그러봐요
# transformer 논문을 벡터디미로 불러와서
# 요약, 인용 등등 할 수 있는 챗봇으로!!

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
from langchain_community.document_loaders import TextLoader, PyPDFLoader

path = './_data/'
pdf_loader = PyPDFLoader(path + "Attention is all you need.pdf")
# pdf_docs = pdf_loader.load()

text_splitter= RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,  # chunk_overlap은 chunk_size보다 작아야함
    separators = ["\n\n", "\n", " ", ""]
)

pdf_split = pdf_loader.load_and_split(text_splitter = text_splitter)


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


########################## 준비완료 #############################

# db = FAISS.from_documents(
#     documents = pdf_split,
#     embedding = embeddings,
# )

DB_PATH = './_db/Faiss19'
# db.save_local(
#     folder_path = DB_PATH,
#     index_name = 'faiss_index19'
# )

db = FAISS.load_local(
    folder_path = DB_PATH,
    index_name = 'faiss_index19',
    embeddings = embeddings,
    allow_dangerous_deserialization= True,
)


############### Retrievers ############### 
############### 검색기 ############### 
retriever = db.as_retriever(search_kwargs={"k":2})
print(retriever)


####################### 모델 연결 #########################

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5.6-terra',
    temperature = 0,
    max_tokens = 1000,  # 너무 적으면 답이 안나오는 경우도 있음, default는 무제한
    api_key = api_key,
    base_url = base_url,
)

# response = model.invoke("nvida가 집중하는 부분?")  # model.invoke는 model.predict와 동일
# print("model의 답변 : ", response.content)
print("====================================================")


from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""

컨텍스트 : {context}     

질문 : {input}

답변 : 
""")   # {context}는 model과 prompt를 이용해서 알아서 생성됨

# 체인 만들기 (아래 두개 체인은 세트로 쓰임)
docu_chain = create_stuff_documents_chain(model, prompt)  # prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) # 검색 | docu_chain



import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들자
demo = gr.ChatInterface(fn = answer_invoke, title = '트랜스포머 FAISS!!')

# Gradio 실행
demo.launch()