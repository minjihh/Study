from sklearn.datasets import load_diabetes    # import 후에 ctrl + space 하면 어떤 리스트들이 있는지 확인 가능
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


#. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)
print(x_train.shape, y_train.shape) # (309, 10) (309,)


#2. 모델 구성

model =Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일 훈련

model.compile(loss= 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_spit=0.5)


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)


#### plot 그리기 ####

import matplotlib.pyplot as plt

