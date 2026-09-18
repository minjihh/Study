from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
import numpy as np
import datetime

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48]))


y = pd.get_dummies(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)   # (124, 13) (54, 13)


x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)



#2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=13, activation='relu'))

# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(3, activation='softmax'))


# input1 = Input(shape = (13,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(40, activation='relu')(dense1)
# drop1 = Dropout(0.5)(dense2)
# dense3 = Dense(40, activation='relu')(drop1)
# drop2 = Dropout(0.5)(dense3)
# dense4 = Dense(40, activation='relu')(drop2)
# drop3 = Dropout(0.5)(dense4)
# dense5 = Dense(40, activation='relu')(drop3)
# drop4 = Dropout(0.5)(dense5)
# dense6 = Dense(40, activation='relu')(drop4)
# drop5 = Dropout(0.5)(dense6)
# output1 = Dense(3, activation='softmax')(drop5)

# model = Model(inputs = input1, outputs = output1)



model = Sequential()
model.add(Conv2D(64, (1,1), input_shape=(13,1,1,), padding='same')) 
model.add(Conv2D(filters=32, kernel_size=(2,1), activation = 'relu', strides = 1, padding = 'same')) # (24,24,32)
model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,1), activation = 'relu', strides = 1, padding = 'same'))  # (23, 23, 32)
# model.add(Conv2D(16, (2, 2), activation = 'relu'))  # (22, 22, 16)
# model.add(Dropout(0.2))   # dropout의 파라미터의 개수 없음
# model.add(Conv2D(16, (2, 2), activation = 'relu'))  # (21, 21, 16)
# model.add(Dropout(0.2))   # dropout의 파라미터의 개수 없음
# model.add(Conv2D(16, (2, 2), activation = 'relu'))  # (20, 20, 16)

model.add(Flatten())
model.add(Dense(units=32, activation = 'relu'))  # units
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape = (32,), activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))  


model.summary()

#3. 컴파일, 훈련

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience =10,
    restore_best_weights=True
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras34/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k34_dropout_wine", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2,  
                #  callbacks = [es, mcp]
                 ) 
end_time = time.time()

# 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss[0])
print("acc: ", round(loss[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score

accuracy_score = accuracy_score(y_test,y_pred)
print('acc_score: ', accuracy_score)

print("걸린시간 (wine data): ", round(end_time - start_time,2), "초")
# cpu 걸린시간 (wine data):  1.74 초
# gpu 걸린시간 (wine data):  1.5 초


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# x 전체가지고 스케일링: 0.8050764858994133
# x_train만 가지고 MinMax 스케일링: 0.4714045207910317
# x_train만 가지고 Standard 스케일링: 0.23570226039551584
# x_train만 가지고 MaxAbs 스케일링: 0.3333333333333333
# x_train만 가지고 Robust 스케일링: 

# acc = 0.95


# acc_score:  0.9259259259259259
# 걸린시간:  2.84 초
# RMSE : 0.3600411499115478

# dropout 적용후
# acc_score:  0.37037037037037035
# 걸린시간:  2.0 초
# RMSE : 1.0363754503432017

# cnn에서 돌렸을때
# acc_score:  0.9814814814814815
# 걸린시간 (wine data):  2.16 초
# RMSE : 0.13608276348795434