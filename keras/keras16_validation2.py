# 시계열은 데이텅 연속되게 잘라야함

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

#1. 데이터

x = np.array(range(1,17))
y = np.array(range(1,17))

# 실습 8개, 4개, 4개 잘라봅시다!!!

x_train = x[:8]
x_val = x[8:12]
x_test = x[12:]

y_train = y[:8]
y_val = y[8:12]
y_test = y[12:]

print(x_train.shape, x_val.shape, x_test.shape)  # (8,) (4,) (4,)


model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_val, y_val))

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)