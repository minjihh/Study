from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time

datasets = fetch_california_housing
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)

print(x_train.shape, y_train.shape)

model=Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_val, y_val))

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)




print(hist.history)

import matplotlib.pyplot as plt

plt.grid()

plt.plot(hist.history['loss'][2:], c = 'red', label = 'loss' )
plt.plot(hist.history['val_loss'][2:], c = 'blue', label = 'val_loss')


plt.legend('updder right')

