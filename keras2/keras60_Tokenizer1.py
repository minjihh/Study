from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()  # 파이썬에서 대문자로 시작하는 부분 -> 클래스 = 객체(인스턴스)
# 토크나이저에서 토큰을 인스턴스로 했다. 
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
# 가장 많이 쓰인 토큰부터 순서대로 정리됨
# 빈도수가 같다면 먼저 등장한 부분을 앞번호로 부여

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = token.texts_to_sequences([text])   # x로 인스턴스화 했다.
print(x)
# [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 9]]   -> 수치화 완료. 임베딩은 아직안됨
# (1, 14)
# 원핫인코딩이 필요함 (2번토큰이 1번토큰의 2배가 아님)

"""
# 원핫 인코딩 3가지 만들기 #
"""


#1. pandas
# import pandas as pd
import numpy as np

# x = np.array(x)
# x = x.reshape(-1)
# x = pd.get_dummies(x, dtype=int)

# print(x)
# print(x.shape)


#2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse_output=False)   # sparse_output=False 설정 부분 잊지 않도록 주의

# x= np.array(x)
# print(x.shape)  # (1,13)

# x = x.reshape(-1,1)
# print(x.shape)
# print(x)

# ohe = ohe.fit(x)   # x데이터 reshape 후에 fit
# x = ohe.transform(x)
# print(x)


#3. keras (현재 tokenizer 데이터는 0값이 없으므로 이부분 주의)
from tensorflow.keras.utils import to_categorical

x = to_categorical(x)

print(x)
print(x.shape)



