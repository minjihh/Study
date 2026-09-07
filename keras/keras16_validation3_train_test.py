# 데이터 16-2 카피

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array(range(1,17))
y = np.array(range(1,17))


# [실습] train_test_split로 잘라요!!

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)


print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델구성

model=Sequential()
model.add((Dense(1, input_dim=1)))
model.add(Dense(1))

#3.

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_val, y_val))


#4.

loss = model.evaluation(x_test, y_test)
results = model.predict(x_test)


