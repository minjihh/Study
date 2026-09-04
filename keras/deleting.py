import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd


path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(train_csv.shape)   # (1459, 10)
print(train_csv)

submission = pd.read_csv(path + "submission.csv", index_col=0)

print(submission)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

train_csv = train_csv.dropna()
print(train_csv)


x = train_csv.drop(['count'], axis = 1)
print(x)

y = train_csv['count']
print(y)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1)


model = Sequential()
model.add(Dense(1, input_dim=10))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=4)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)