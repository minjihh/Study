# LangGraph: State / Node / Edge
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyDate(TypedDict):
    messages : Annotated[list, add_messages]   # Reducer: list를 append -> add_messages가 reducer 역할

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content = '안녕', id = '1')]
msg2 = [AIMessage(content = '방가', id = '2')]

results = add_messages(msg1, msg2)
print(results)


########################################################
msg3 = [HumanMessage(content = '안녕', id = '3')]
msg4 = [AIMessage(content = '방가', id = '3')]

results2 = add_messages(msg3, msg4)
print(results2)  # id가 동일하면 덮어쓰게 된다.