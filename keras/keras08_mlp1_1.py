import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense     # from 안에 뭐들었는지 궁금하면 ctrl + space
import numpy as np


#1. 데이터

# x가 (2,5), y는 (5,)라서 아래 데이터로는  학습 불가. x(5,2), y(5,)이거나 x(2,5), y(2,) 이어야 학습 가능
# x = np.array([[1,2,3,4,5],
#               [6,7,8,9,10]])
# y = np.array([1,2,3,4,5])


# 위에 데이터를 올바르게 수정한 아래 데이터. // 열=column=피처
x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]])
y = np.array([1,2,3,4,5])


print(x.shape)  # (5,2)
print(y.shape)  # (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 행무시 열우선: column개수 확인해서 input_dim 설정

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epohcs=100, batch_size= 3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
results = model.predit(np.array[[6,11]])       # shape: (1,2)  // predict시에도 행무시 열우선
print("[6,11]의 예측값 :", results)
