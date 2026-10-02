# text가 2개일때
# x = token.texts_to_sequences([text1, text2]) 적용후
# np.concatenate(x) 적용


from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘 생겼다.'

token = Tokenizer()  # 파이썬에서 대문자로 시작하는 부분 -> 클래스 = 객체(인스턴스)
# 토크나이저에서 토큰을 인스턴스로 했다. 
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9, '개똥이는': 10, '기관사를': 11, '좋아한다': 12, '말똥이는': 13, '잘생겼다': 14, '길동이는': 15, '더': 16, '잘': 17, '생겼다': 18}


# numpy의 concatenate 이용

x = token.texts_to_sequences([text1, text2])
x = np.concatenate(x)  # 두개의 text concat한 뒤에 one hot 인코딩 적용

print(x)

"""
# 원핫 인코딩 3가지 만들기 #
"""


#1. pandas
# import pandas as pd


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