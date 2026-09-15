from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import time
import datetime


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
# model.add(Dropout(0.5))

# model.add(Dense(40, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(10, activation = 'softmax'))


input1 = Input(shape = (64,))
dense1 = Dense(5)(input1)
dense2 = Dense(40, activation='relu')(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(40, activation='relu')(drop1)
drop2 = Dropout(0.5)(dense3) 
dense4 = Dense(40, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense4)
dense5 = Dense(40, activation='relu')(drop3)
drop4 = Dropout(0.5)(dense5)
output1 = Dense(10, activation='softmax')(drop4)

model = Model(inputs = input1, outputs = output1)
model.summary()

es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=10,
    restore_best_weights=True
)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras34/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k34_dropout_digits", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_split=0.2, 
                #  callbacks=[es, mcp]
                 )
end_time = time.time()

loss = model.evaluate(x_test,y_test)
print("loss: ", loss[0])
print("acc: ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", acc_score)
print("걸린시간 (digits data): ", round((end_time - start_time),2), "초")
# cpu 걸린시간 (digits data):  3.94 초
# gpu 걸린시간 (digits data):  6.94 초

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

# dropout 적용후
# acc_score:  0.362962962962963
# 걸린시간:  14.97 초
# RMSE : 3.4321465046861896

# 함수형
# acc_score:  0.1259259259259259
# 걸린시간:  5.94 초
# RMSE : 3.307399112427904