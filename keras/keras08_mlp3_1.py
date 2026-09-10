# x.shape와 y.shape에서 (n,)에서 n값이 다를경우 전치나 또다른 데이터 처리가 필요
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1,]]).transpose()
print(x.shape, y.shape) # (3,10) (2,10) 전치후 -> (10,3) (10,2)



# [실습]
# [10,31,211]의 예측값
# 요구사항 11.00, 0.00
#2. 모델 구성

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(2))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=500, batch_size=2)

#4. 평가, 예측

loss = model.evaluate(x,y)
print('loss: ', loss)
results = model.predict(np.array([[10,31,211]]))
print('results: ', results)