# 54-1 카피
# Conv1D -> 생각보다 잘 통함 / 시계열일때 Convolution

### input data ###
# DNN: 2차원데이터
# RNN: 3차원 데이터
# CNN: 4차원 데이터

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import Conv1D, Flatten, GlobalAveragePooling1D

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
print(x)



#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3, 1)))
# model.add(SimpleRNN(3, input_shape = (3, 1)))  # activation tanh 적용됨
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능

model.add(Conv1D(filters = 10, kernel_size =2, input_shape = (3,1)))  # Conv1D에서 kernel size는 시계열에서 timestemp와 비슷한 의미
# Conv1D -> 3차원 입력, 3차원 출력 -> Dense와 연결하기위해 Flatten이나 GAP 필요
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(10, activation = 'relu'))  # 중간에 activation sigmoid 써도 문제없음. 성능이 좋다면 쓰면 됨
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.summary()
# exit()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 3000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)

# 10.9999999