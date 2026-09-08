# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# path = "./_data/kaggle_santander/"
path = "c:/study/_data/kaggle_santander/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)


print(train_csv.shape)  # (200000, 202)
print(test_csv.shape)  #  (200000, 201)
print(submission_csv.shape)  #  (200000, 2)

# 결측치 확인
print(train_csv.info())   # 방법1 (지금 santander 데이터는 데이터 개수가 너무 많아서 이방법으로 확인 힘듦)
print(train_csv.isna().sum())  # 방법2
print(train_csv.isnull().sum())  # 방법3

x = train_csv.drop(['target'], axis = 1)
y = train_csv['target']
print(x.shape, y.shape)  # (200000, 200) (200000,)

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 1, stratify=y)

model = Sequential()
model.add(Dense(5, input_dim = 200))
model.add(Dense(9))
model.add(Dense(15))
model.add(Dense(1))


model.compile(loss='binarycrossentropy', optimizer = 'adam',)
hist = model.fit(x_train, y_train, epochs = 10000, batch_size=16, validation_split = 0.3, metrics=['acc'])

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = np.round(model.predict(x_test))

Accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", Accuracy)






