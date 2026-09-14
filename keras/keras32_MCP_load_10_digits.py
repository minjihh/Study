from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import time
import datetime

pt_path = './_save/keras31/'


datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = pd.get_dummies(y, dtype = int)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# model = Sequential()
# model.add(Dense(5, input_dim = 64, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(10, activation = 'softmax'))

model = load_model(pt_path + 'k30_0914_1442-0014-0.3713.keras')

es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True
)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k30_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=100000, batch_size=4, validation_split=0.2, callbacks=[es, mcp])
# end_time = time.time()

loss = model.evaluate(x_test,y_test)
print("loss: ", loss[0])
print("acc: ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", acc_score)
# print("걸린시간: ", round((end_time - start_time),2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# 스케일링 없이 실행: 1.9460310986818883
# x_train만 가지고 MinMax 스케일링: 1.7040257344605167
# x_train만 가지고 Standard스케일링: 1.9488838211007213
# x_train만 가지고 MaxAbs 스케일링: 1.2758439472669758
# x_train만 가지고 Robust 스케일링: 

# acc: 1.0

# acc_score:  0.8888888888888888
# 걸린시간:  8.11 초
# RMSE : 1.6815997674172585