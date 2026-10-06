# 예나를 CNN으로 만드세요
# 맹그러봐

# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

# import os 
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"   # 메모리 모으기, 메모리 이슈있을때 사용
# tensorflow cuda 연결되는 부분이라 tensorflow import 전에 사용
# scaling shape
# datatype을 바꾸면 속도가 훨씬 빨라질 수 있다.
# tabular데이터는 배치사이즈 크게하면 훈련속도 빨라짐 (jena 데이터의 경우 split과 scaling만 진행)

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import time


# wd를 y로 (제일 끝에 있는 column이라 자르기 쉬움)
# 2016년 12월 31일의 하루치 wd 예측
# column은 14개 뒤에서부터 144

#1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets.info())
print(datasets.shape)  # (420551, 14)

# y_cor = datasets[-144:]['wd (deg)']  # 예측치 정답 데이터
y_cor = datasets[-144:]['T (degC)']  # 예측치 정답 데이터

print(y_cor.shape)  # (144,)
y_cor = y_cor.values.reshape(-1,1)





### 훈련할 데이터 자르기 ###
# x_data = datasets[:-288].drop(['wd (deg)'], axis =1).to_numpy(dtype= np.float32)
# y_data = datasets[144:-144]['wd (deg)'].to_numpy(dtype=np.float32)

x_data = datasets[:-288].drop(['T (degC)'], axis =1).to_numpy(dtype= np.float32)
y_data = datasets[144:-144]['T (degC)'].to_numpy(dtype=np.float32)


print(x_data.shape)  # (420263, 13)
print(y_data.shape)  # (420263,)




### x 데이터 4차원으로 reshape 


x_data = x_data.reshape(-1,13,1,1)

print(x_data.shape)  # (420263, 13, 1, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data,
    train_size = 0.7,)


# scaling 하기전에 2차원 데이터로 reshape 필요
x_train_scaler = x_train.reshape(-1, x_train.shape[-1])
x_test_scaler = x_test.reshape(-1, x_test.shape[-1])

# scaling은 train 나눈후에 train만 이용해서
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train_scaler)    # sklearn에서 fit은 보통 실행하라는 의미
x_train = scaler.transform(x_train_scaler)
x_test = scaler.transform(x_test_scaler)

x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

print(x_train.shape, y_train.shape)  # (294084, 144, 13) (294084,)
 
 
# submit용 x데이터
# x_predict = datasets[-288:-144].drop(['wd (deg)'], axis =1)
x_predict = datasets[-288:-144].drop(['T (degC)'], axis =1)

print(x_predict.shape) # (144, 13)

x_predict = np.array(x_predict)
x_predict = x_predict.reshape(-1, 1)
print(x_predict.shape)


x_predict = scaler.transform(x_predict)

# pandas dataframe reshape
# x_predict = x_predict.values.reshape(-1,144,13)  
# # reshape 두번째 방법
# x_predict = x_predict.to_numpy()
x_predict = x_predict.reshape(-1,13,1,1)
print(x_predict.shape)  # (144, 13, 1, 1)


#2. 모델구성

model= Sequential()
model.add(Conv2D(10, (13,1), input_shape = (13,1,1), padding='same'))
model.add(Conv2D(filters=32, kernel_size=(13,1), activation = 'relu', strides = 1, padding = 'same')) # (24,24,32)


model.add(Flatten())
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### mcp 파일명 ###
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras66/'
filename = '{epoch:04d}-{val_loss:.4f}_jena.keras'
filepath = "".join([pt_path, "k66_", date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    save_best_only=True,
    verbose=1
)

learning_rate = 0.01
model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    restore_best_weights= True,
    verbose = 1
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)


start_time = time.time()
model.fit(x_train, y_train,
          epochs = 50,
          batch_size = 600,    # keras batch size default = 32
        #   validation_split=0.2,  # shuffle 없이 뒤에서 부터 해당 비율을 잘라 validation으로 이용, validation_data와 validation_split 동시에 있을경우 validation_data가 override
          validation_data = (x_test, y_test),
          verbose =1,
        callbacks = [es, rlr])
end_time = time.time()



#4. 평가 예측


y_pred = model.predict(x_predict)

print("걸린시간: ", round(end_time - start_time, 2))

# print("x_predict의 예측값: ", y_pred)
loss = model.evaluate(x_predict, y_cor)
print("loss: ", round(loss, 2))


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_cor, y_pred)
print("RMSE :", rmse)

# T (degC) 예측
# 10epochs
# loss:  12.49
# RMSE : 3.534246079840953

# 50 epochs patience 10
# loss:  12.25
# RMSE : 3.499566659838593

# with CNN
# loss:  145.48
# RMSE : 12.061416631951818