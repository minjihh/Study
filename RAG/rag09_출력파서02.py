# 9-1 카피
# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser
# parser: 분석하다. 답변을 다듬다.

# 페르소나 부여 -> 프롬프트로 역할을 줌
template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 
주어진 상황에 맞는 영어회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

#상황:
{question}

#FORMAT:
- 영어회화 : 
- 한글번역 :
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

prompt = PromptTemplate.from_template(template = template)

model = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature = 0,
    api_key = api_key,
    base_url= base_url,
)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# output_parser를 포함하면 content가 없어진다.
chain = prompt | model | output_parser

input = {"question" : "재밌는 미드 추천해줘"}

# invoke가 model.predict과 같은 기능
response = chain.invoke(input)
print(response)


