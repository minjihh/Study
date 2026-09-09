from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd

datasets = fetch_covtype()
print(fetch_covtype.info())

x = datasets.data
y = datasets.target
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

# one hotencoding


x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7, random_state=1)

model=Sequential()
model.add(Dense(5, input_dim= , activation='relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(, activation = ''))

model.compile(loss='', optimizer='adam')
hist = model.fit(x_train, y_train, epochs= 100000, batch_size = 128, validation_split=0.2)


# 시간재기
# 배치 크게
# acc 0.93