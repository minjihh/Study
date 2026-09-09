import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_iris()
print(datasets)
print(datasets.DESCR)  # 판다스의 describe 있음
print(datasets.feature_names) # 판다스의 ".columns" 똑같은 기능

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))


x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7, random_state=1,
                                                  )

model = Sequential()
model.add(Dense(5, input_dim=4, activation = 'relu'))
model.add(Dense(40, activateion='relu'))
model.add(Dense(3, activatoin = 'sigmoid'))


model.compile(loss='categrocial_crossengropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=10, batch_size=16, callbacks = [es], validation_split=0.3)




loss = model.eveluate(x_val,y_val)
y_pred = model.predict(x_test)
