# rag02 카피

from langchain_openai import ChatOpenAI
import os
# 내컴퓨터 윈도우의 환경변수에 api key를 넣음 (일회성으로 한번실행하고 주석처리하면 실행안됨)
# os.environ["OPENAI_API_KEY"] = ""키입력"


llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

# chatgpt에 '안녕하세요' 입력한것과 같은 효과
# 지금은 메모리 기능이 없어서 이전에 입력된 내용도 기억하지 못함
# 나중에 agent 구성할때는 메모리 기능을 추가해줘야 함
# response = llm.invoke('안녕하세요. 나는 윤영선이야')
response = llm.invoke('나는 누구게?')

# print(response)
print(response.content)



