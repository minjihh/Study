import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer  # 유방암관련 데이터셋 불러오기


datasets = load_breast_cancer()

x = datasets.data
x = datasets['data']
y = datasets.target

print(datasets.DESCR)
print(datasets.feature_names)

print(np.unique(y))
print(np.unique(y, return_counts=True))

print(pd.DataFrame.value_counts())
print(pd.Series.value_counts())

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True
)

model.compile(loss = 'BCE', optimizer = 'adam',
              metrics=['acc'])
hist = model.fit(x_train, y_train, epochs =10, batch_size=16, validatoin_data = (x_val, y_val), callbacks = [es])

print(hist.history)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))

plt.rcParams['font.family'] = 'Malgun Gothic'

plt.plot(hist.history['loss'], c = 'red', label ='loss')
plt.plot(hist.history['val_loss'], c = 'red', label ='val_loss')

plt.title('breast cancer loss')

plt.legend(loc = 'upper right')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.grid()
plt.show()