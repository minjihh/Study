# ohe 적용 #
# 이 파일에서 사용하는 데이터는 개수가 많지 않아 ohe를 적용해도 큰 문제없음
# 하지만 데이터가 많을때 ohe를 하면 0이 지나치게 많아지는 문제가 생김 -> Embedding으로 해결
# -> x 쪽에서 ohe 하는건 큰 문제가 될 수 있음
# ohe하는 이유: 남자 label 2 여자 label 1 일때 남자가 여자의 2배가 아니므로 위치값으로 바꿔주기 위함

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


# LSTM 적용위해 (15, 5, 1)로 만들어줘야함
print(padded_x.shape)  # (15, 5) 

## 원핫인코딩 ##
# 2. sklearn
from sklearn.preprocessing import OneHotEncoder    # ohe는 reshape후에 ohe.fit_transform()
ohe = OneHotEncoder(sparse_output=False)   # sparse_output=False 설정 부분 잊지 않도록 주의

padded_x= np.array(padded_x)
print(padded_x.shape)  # (1,13)
print(padded_x)


padded_x = padded_x.reshape(-1,1)
print(padded_x)
print(padded_x)

ohe = ohe.fit(padded_x)   # x데이터 reshape 후에 fit
ohe_padded_x = ohe.transform(padded_x)
print(ohe_padded_x)

print(ohe_padded_x.shape) # (75, 31)

ohe_padded_x = np.array(ohe_padded_x)
ohe_padded_x = ohe_padded_x.reshape(15,5,31)


x_train, x_test, y_train, y_test = train_test_split(ohe_padded_x, labels,
                                                    train_size=0.7,
                                                    random_state =1,
                                                    )

x_predict = ['개똥이 잘생겼다']
x_predict = token.texts_to_sequences(x_predict)

print(x_predict)

padded_x_predict = pad_sequences(x_predict,
                                padding = 'pre',
                                maxlen= 5,
                                truncating = 'post'
                                )

print(padded_x_predict)
print(x_train.shape, y_train.shape)  # (10, 5) (10,)


print(padded_x_predict.shape)  # (1, 5)

padded_x_predict = padded_x_predict.reshape(-1, 1)
ohe_padded_x_predict = ohe.transform(padded_x_predict)
print(ohe_padded_x_predict.shape)  # (5, 31)
print(ohe_padded_x_predict) 

# # pad_sequences로인해 기존의 token.word_index에 없던 0이라는 값이 생긴 상태.
# # 새로생긴 0값을 포함하여 원핫인코딩된 값에서 0번째 columnd은 어차피 의미가 없는 부분이므로 제거하고 싶담녀 아래와 같이 적용
# ohe_padded_x_predict = ohe_padded_x_predict[:,1:]
# print(ohe_padded_x_predict) 





ohe_padded_x_predict = ohe_padded_x_predict.reshape(1,5,31)



# 맹그러봐 DNN
#2. 모델구성

model = Sequential()
model.add(LSTM(10, input_shape= (5,31)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = learning_rate),
              metrics = ['acc'])

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 10,
          batch_size = 64,
          validation_data = (x_test, y_test),
          verbose = 1,
          )
end_time = time.time()


#4. 예측 평가

print("걸린시간: ", round(end_time - start_time, 2))
loss = model.evaluate(x_test, y_test)
print("loss: ", loss[0])
print("acc: ", loss[1])


y_pred = model.predict(ohe_padded_x_predict)
y_pred = np.round(y_pred)

print("예측: ", y_pred)


y_cor = [1]
y_cor = np.array(y_cor)

acc_score = accuracy_score(y_cor, y_pred)
print("accuracy_score: ", round(acc_score, 2))

# 예측:  [[1.]]
# accuracy_score:  1.0
