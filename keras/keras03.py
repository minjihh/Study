# 데이터가 많아질 수록 성능이 좋아짐
# 훈련을 많이 시켰을때는 과적합이 발생할 수 있으나 데이터 자체는 많으면 많을수록 좋다. (좋은데이터라는 가정하에)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss )
result = model.predict(np.array([1,2,3,4,5]))
print("7의 예측값 : ", result)