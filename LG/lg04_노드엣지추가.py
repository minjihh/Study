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
from langchain_core.messages import AIMessage, HumanMessage

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
def prepare(state: State):
    return{"messages" : 
           [AIMessage(content="질문을 접수했습니다.")]
           }

def chatbot(state : State): 
    return{'messages' : [llm.invoke(state["messages"])]}

def finish(state : State):
    return{"messages" : 
        [AIMessage(content="답변 처리가 완료되었습니다.")]
        }


###################### 3. 그래프 구성 ######################
graph_01 = StateGraph(State)   # model = Sequential()

graph_01.add_node("prepare", prepare)
graph_01.add_node("chatbot", chatbot)
graph_01.add_node("finish", finish)


###################### 4. 엣지 추가 ######################
graph_01.add_edge(START, "prepare")
graph_01.add_edge("prepare", "chatbot")
graph_01.add_edge("chatbot", "finish")
graph_01.add_edge("finish", END)

###################### 5. 그래프 컴파일 ######################
graph = graph_01.compile()

###################### 6. 그래프 실행 ######################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

###################### 7. 결과 출력 ######################
for i,msg in enumerate(response['messages']):
    print(f"[{i}] {msg.type} : {msg.content}")

# [3] ai :답변 처리가 완료되었습니다.


