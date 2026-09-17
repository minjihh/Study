from sklearn.datasets import load_diabetes    # import 후에 ctrl + space 하면 어떤 리스트들이 있는지 확인 가능
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time

path = './_save/keras34/'

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)
print(x_train.shape, y_train.shape) # (309, 10) (309,)

### robust scaling ###
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# from sklearn.preprocessing import RobustScaler

# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler = scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


### 스케일링 1 MinMax
# mnist -> minmax scaling이 적절. maxabs도 동일
x_train = x_train/255.   # .을 붙임으로서 float 형태로 출력하게 됨
x_test = x_test/255.

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0


print(x_train.shape, x_test.shape)  # (309, 10) (133, 10)

x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)
print(x_train.shape, x_test.shape) # 


#2. 모델 구성

# model =Sequential()
# model.add(Dense(5, input_dim=10))
# model.add(Dropout(0.5))

# model.add(Dense(9))
# model.add(Dropout(0.5))

# model.add(Dense(1))


# input1 = Input(shape = (10,))
# dense1 = Dense(5, activation='relu')(input1)
# # drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(9, activation='relu')(dense1)
# # drop2 = Dropout(0.5)(dense2)
# output1 = Dense(1)(dense2)

# model = Model(inputs=input1, outputs=output1)

model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(5,2,1,), padding='same')) 
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
model.add(Dense(1))  

model.summary()



#3. 컴파일 훈련

model.compile(loss= 'mse', optimizer= 'adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True
)


### mcp 파일명 ###
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k34_dropout_diabetes", date, "-", filename])



mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    save_best_only=True,
    verbose=1
)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_split=0.5,
                #   callbacks = [es, mcp]
                  )

end_time = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")
# cpu 걸린시간:  1.6 초
# gpu 걸린시간:  2.12 초


print("mse: ", loss)

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# 스케일링 없이 실행: 142.60912583270098
# x_train만 가지고 MinMax 스케일링: 96.7204834289024
# x_train만 가지고 Standard스케일링: 154.41742585859953
# x_train만 가지고 MaxAbs 스케일링: 153.33504717555243
# x_train만 가지고 Robust 스케일링: 


# mse:  21896.89453125
# r2:  -3.34939535870292
# RMSE : 147.9759999433259

# drop out 적용
# mse:  25322.9609375
# r2:  -4.029916806327675
# RMSE : 159.13189737650825

# 함수형
# mse:  26087.681640625
# r2:  -4.181814258010112
# RMSE : 161.5168220040415

# cnn으로 
# 걸린시간:  3.85 초
# mse:  5029.4794921875
# r2:  0.0009911675254793417
# RMSE : 70.9188205119668