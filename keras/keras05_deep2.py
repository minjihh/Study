#### model.add를 더 간략하게 나타내기 
# keras04에서는 인풋노드와 아웃풋 노드 개수를 다 적어줬지만 keras 05에서 간략하게 표기
# 첫번재 layer에서만 input_dim 표기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,4,3,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss )
# result = model.predict(np.array([1,2,3,4,5]))
# print("7의 예측값 : ", result)