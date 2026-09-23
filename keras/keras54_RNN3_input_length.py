# 54-1 카피

### input data ###
# DNN: 2차원데이터
# RNN: 3차원 데이터
# CNN: 4차원 데이터

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

# 10, 벡터데이터를 (7,3) 매트릭스로 바꿈

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)  # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)  # (7,3,1)
print(x.shape)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3, 1)))
# model.add(SimpleRNN(10, input_shape = (3, 1)))
model.add(SimpleRNN(units=10, input_length=3, input_dim=1))  # input_shape를 썼을 때와 동일, 파라미터화 되어 있으므로 input_length와 input_dim 순서 바뀌어도 상관 없음 # encoding 할때 유용
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(10, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))


model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 100)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)

