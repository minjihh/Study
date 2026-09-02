# import ssl
# ssl._create_default_https_context = ssl._create_unverified_

from sklearn.datasets import fetch_california_housing    # import 후에 ctrl + space 하면 어떤 리스트들이 있는지 확인 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터
datasets = fetch_california_housing()  # 소문자로 시작하고 뒤에 () 붙으므로 함수임을 짐작할 수 있음
x = datasets.data
y = datasets.target
print(x.shape, y.shape)    # (20640, 8)  (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=4)

#4. 평가, 예측
print("=================================================")
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
results = model.predict(x_test)
print("results: ", results)


