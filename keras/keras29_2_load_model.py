# 29-1 카피

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    random_state=1,
                                                    stratify=y)

print(x_train.shape, y_train.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim = 8, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activatoin = 'relu'))
model.add(Dense(40, activatoin = 'relu'))
model.add(Dense(40, activatoin = 'relu'))
model.add(Dense(1))

path = './_save/'
model = load_model(path + 'keras29.keras')

model.summary()

#3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor='val_loss',
                   mode = 'auto',
                   patience=10,
                   restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs =10000,
                 batch_size=16,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es])
end_time = time.time()

model.save(path+'keras29.keras')

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))

rmse = RMSE(y_test, y_pred)

print("rmse: ", rmse)


import matplotlib.pyplot as plt

plt.rcParams['font_family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6))

plt.plot(hist.hitory['loss'][2:], c = 'red', label='loss')
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')

plt.legend(loc='upper right')

plt.title('캘리포니아 로스')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.grid()
plt.show()
