from sklearn.datasets import fetch_california_housing, load_diabetes    # import 후에 ctrl + space 하면 어떤 리스트들이 있는지 확인 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (442,10) (442,)

# [실습] 맹그러봐요!!

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state = 15)


#2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=10))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(15))
model.add(Dense(13))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=16)


#4. 평가, 예측
print("===================================")
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
# results = model.predict(x_test)
# print("results: ", results)