# rag05 카피

from langchain_openai import ChatOpenAI
import os

# .env 파일 만든후에 api key를 입력해논 상태에서 아래와 같이 사용가능
from dotenv import load_dotenv
load_dotenv()

# openAI와 달리 url도 같이 입력을 해줘야 함
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key = api_key,
    base_url= base_url,
)

# chatgpt에 '안녕하세요' 입력한것과 같은 효과
# 지금은 메모리 기능이 없어서 이전에 입력된 내용도 기억하지 못함
# 나중에 agent 구성할때는 메모리 기능을 추가해줘야 함
# response = llm.invoke('안녕하세요. 나는 윤영선이야')
response = llm.invoke('나는 누구게?')

# print(response)
print(response.content)



