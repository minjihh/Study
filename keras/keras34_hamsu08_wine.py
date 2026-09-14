from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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


input1 = Input(shape = (13,))
dense1 = Dense(5)(input1)
dense2 = Dense(40, activation='relu')(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(40, activation='relu')(drop1)
drop2 = Dropout(0.5)(dense3)
dense4 = Dense(40, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense4)
dense5 = Dense(40, activation='relu')(drop3)
drop4 = Dropout(0.5)(dense5)
dense6 = Dense(40, activation='relu')(drop4)
drop5 = Dropout(0.5)(dense6)
output1 = Dense(3, activation='softmax')(drop5)

model = Model(inputs = input1, outputs = output1)
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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2,  callbacks = [es, mcp]) 
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

print("걸린시간: ", round(end_time - start_time,2), "초")

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