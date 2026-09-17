from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 이때 3단계에서 validation_data하면 어떤 데이터가 va/test로 쪼개지는지?

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)


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


print(x_train.shape, x_test.shape)  # (404, 13) (51, 13)

x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)
print(x_train.shape, x_test.shape) # 


#2. 모델 구성
# model =Sequential()
# model.add((Dense(3, input_dim=13)))
# model.add(Dropout(0.5))

# model.add(Dense(1))

# input1 = Input(shape = (13,))
# dense1 = Dense(3)(input1)
# drop1 = Dropout(0.5)(dense1)
# output1 = Dense(1)(drop1)

# model = Model(inputs = input1, outputs = output1)


model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13,1,1,), padding='same')) 
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

path = './_save/keras34/'

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True,
    verbose = 1
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k34_dropout_boston", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    save_best_only=True,
    verbose = 1

)

model.compile(loss='mse', optimizer='adam')

import time

start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=10, 
                batch_size=4, 
                validation_split= 0.2,
                # callbacks=[es, mcp]
                )
# model.fit(x_train, y_train, epochs=10, batch_size=4, validation_split= 0.33)
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2))

# cpu 걸린시간:  1.58
# gpu 걸린시간:  2.5


print("mse: ", loss)

r2 = r2_score(y_test, results)
print("r2: ", r2)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# x 전체로 scaling : # RMSE : 8.672873642737752
# x_train만으로 MinMax scaling : RMSE : 12.013473382237372
# x_train만 가지고 Standard스케일링: 20.39037099368951
# x_train만 가지고 MaxAbs 스케일링: 11.12472809697239
# x_train만 가지고 Robust 스케일링: 


# mse:  611.0635375976562
# r2:  -5.868082091195529
# RMSE : 24.719699338997366

# drouout 적용후
# mse:  555.52685546875
# r2:  -5.243874965539267
# RMSE : 23.5696180259663

# 함수형
# mse:  639.330322265625
# r2:  -6.18578875479183
# RMSE : 25.284983140707936

# cnn으로 
# 걸린시간:  4.45
# mse:  61.30900955200195
# r2:  0.3109140018687312
# RMSE : 7.830007141505531