# R2 기준 0.55 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


# 1.데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
(x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.7, random_state=10)
print(x_train.shape, y_train.shape)  # (14447,8) (14447,)


#2. 모델 구성

model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=4)


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
# print("results: ", results)