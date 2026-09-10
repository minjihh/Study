from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import time


datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(5, input_dim = 64, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100000, batch_size=4, validation_split=0.2, callbacks=[es])
end_time = time.time()

loss = model.evaluate(x_test,y_test)
print("loss: ", loss[0])
print("acc: ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", acc_score)
print("걸린시간: ", round((end_time - start_time),2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# x 전체가지고 스케일링: 
# x_train만 가지고 스케일링: 


# acc: 1.0