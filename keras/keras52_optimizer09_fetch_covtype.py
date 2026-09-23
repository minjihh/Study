from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import numpy as np
import pandas as pd

datasets = fetch_covtype()
# print(fetch_covtype.info())

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

# exit()

# one hotencoding
y = pd.get_dummies(y)
# one hotencoding to_categorical로 할경우 0부터 생성되므로 나중에 argmax할때 +1하는 작업을 해줘야함


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model=Sequential()
model.add(Dense(5, input_dim=54, activation='relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

es = EarlyStopping(
    monitor = 'val_loss',
    mode='auto',
    patience = 50,
    restore_best_weights=True
)


from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

start_time = time.time()
hist = model.fit(x_train, y_train, epochs= 100000, batch_size = 128, validation_split=0.2, callbacks=[es])
end_time = time.time()

loss = model.evaluate(x_train, y_train)
print("loss: ", loss[0])
print("acc: ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
acc_score = accuracy_score(y_test, y_pred)

print("acc_score: ", round(acc_score,2))
print("걸린시간: ", round(end_time - start_time, 2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# 스케일링 없이 실행: 1.0469764005252644
# x_train만 가지고 MinMax 스케일링: 0.8720461431932777
# x_train만 가지고 Standard스케일링:  0.9029299501906831
# x_train만 가지고 MaxAbs 스케일링: 0.7926675107714721
# x_train만 가지고 Robust 스케일링: 

# 시간재기
# 배치 크게
# acc 0.93

# lr 0.01 일때, patience 50
# acc_score:  0.79
# 걸린시간:  388.49 초
# RMSE : 1.099105587107108