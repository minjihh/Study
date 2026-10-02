# 61-3 카피
# ohe 적용 #
# Embedding: vector화 -> 이후 cosine 유사도와 같은 방법을 이용해서 유사도 측정

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',

]


labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]) # 긍정 데이터는 1, 부정데이터는 0으로 라벨링된 상태

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, 
# '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14,
# '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
# '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, 
# '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}


x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], 
# [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]

################### 패딩 ################### 
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding = 'pre',  # pre: 앞쪽을 0으로 채움, post: 뒤쪽을 0으로 채움
                        maxlen = 5,  # maxlen보다 길경우: default는 앞쪽부터 잘려나감
                        truncating = 'post'  # maxlen보다 길때 어디부터 자를지 결정
                         )


print(padded_x)
print(padded_x.shape)  # (15, 5)

exit()

#2. 모델
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential() 
########### 임베딩1 #############
# model.add(Embeding(input_dim=30, output_dim=100, input_length=5))
# # input_dim: 단어d사전의 갯수, output_dim: 차원, input_length: Length of input sequences
# input_length 값에 행무시 "열우선" 적용
# Enbedding layer이후 시계열 게열 모델에 들어가기 때문에 3차원 형태로 출력
# model.add(SimpleRNN(10))
# model.add(Dense(1))


"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 5, 100)            3000      
                                                                 
 simple_rnn (SimpleRNN)      (None, 10)                1110  
"""
########### 임베딩2 #############
# model.add(Embedding(input_dim=30, output_dim=100))  # input_length 명시안해도 알아서 맞춰줌
# # input_dim: 단어사전의 갯수, output_dim: 차원, input_length: Length of input sequences
# 30 * 100 크기의 단어 사전에거 값을 원본데이터로 치환
# model.add(SimpleRNN(10))
# model.add(Dense(1))


########### 임베딩3 #############
model.add(Embedding(30, 100,))  # 임베딩레이어에서는 앞에나오는 숫자가 인풋레이어 (다른 레이어에서는 앞에나오는 숫자가 아웃풋 숫자였음)
# model.add(Embedding(30, 100, 5))  # input_dim, output_dim, input_length 안된다! (input_length부분은 자동으로 잡아주므로 차라리 적지 않는게 나음)
# model.add(Embedding(30, 100, input_length=5))  # 된다!!!

model.add(SimpleRNN(10))
model.add(Dense(1))


model.summary()

#3. 컴파일 훈련

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['acc'])
model.fit(padded_x, labels, epochs =3 )









