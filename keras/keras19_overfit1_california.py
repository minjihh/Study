# 17-1 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)

print(x_train.shape, y_train.shape)   # (14447, 8) (14447,)

#2. 모델구성
model =Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1, validation_data = (x_val, y_val)
        #   valdiatoin_split=0.5
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print("===================== history ========================")
print(hist)
print("===================== history ========================")






