from langchain_openai import ChatOpenAI
import os

# .env 파일 만든후에 api key를 입력해논 상태에서 아래와 같이 사용가능
from dotenv import load_dotenv
load_dotenv()

# openAI와 달리 url도 같이 입력을 해줘야 함
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    # model_name = 'gpt-5.6-terra',
    model_name = 'gpt-5-nano',
    temperature=0,
    max_tokens = 1000,
    api_key = api_key,
    base_url= base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

###################### 1. State 정의 ###################### 
class State(TypedDict):
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : int


# 위에꺼는 한마디로 아래다! 
# state = {
#     "messages" : [],
#     'aaa' : 'abc',
#     'bbb' : 'abc',
#     'ccc' : 43,
# }


###################### 2. node 정의 ######################
def chatbot(state : State): 
    return{'messages' : [llm.invoke(state["messages"])]}


###################### 3. 그래프 구성 ######################
graph_01 = StateGraph(State)   # model = Sequential()

graph_01.add_node("chatbot", chatbot)

###################### 4. 엣지 추가 ######################
graph_01.add_edge(START, "chatbot")

graph_01.add_edge("chatbot", END)

###################### 5. 그래프 컴파일 ######################
graph = graph_01.compile()

###################### 6. 그래프 실행 ######################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

# print(response)
print(response["messages"][0])
print(response["messages"][1])
print(response["messages"][1].content)
print(response["messages"][0].type)  # human
print(response["messages"][1].type)  # ai




