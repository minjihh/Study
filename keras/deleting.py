import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score


from sklearn.datasets import load_iris


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size =0.8,
                                                    random_state=1,
                                                    stratify=y)

print(x_train.shape, y_train.shape)
print(np.unique(y, return_counts=True))

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = MinMaxScaler()
x_train = scaler.fit_trainsform(x_train)
x_test = scaler.fit_transform(x_test)

print(x)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=8, activatoin ='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40))
model.add(Dense(1))

model.summary()

path = './_save/keras29/'
model.save(path + 'keras29.keras')


#3. 컴파일, 훈련


es = EarlyStopping(
    monitor='val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True
)

model.compile(loss='mse', optimizer ='adam')

start_time = time.time()
hist = model.fit(x_train, y_train,
              epochs=1000,
              batch_size=16,
              verbose =1,
              validatoin_split=0.3,
              callbacks=[es])
end_time = time.time()


#4. 평가 예측

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred)
    return(np.sqrt(mean_squared_error(y_test, y_pred)))

rmse = RMSE(y_test, y_pred)
print("rmse: ", rmse)

import matplotlib.pyplot as plt

plt.rcPrams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'][2:], c = 'red', label='loss')
plt.plot(hist.history['val_loss'][2:], c= 'blue', label='val_loss')

plt.leged(los= 'upper right')

plt.title('캘리포니아 Loss')

plt.xlabel("epoch")
plt.ylabel("loss")

plt.grid()
plt.show()


