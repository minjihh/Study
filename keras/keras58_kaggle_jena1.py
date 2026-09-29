# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

# import os 
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"   # 메모리 모으기, 메모리 이슈있을때 사용
# tensorflow cuda 연결되는 부분이라 tensorflow import 전에 사용
# scaling shape

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

import time


# wd를 y로 (제일 끝에 있는 column이라 자르기 쉬움)
# 2016년 12월 31일의 하루치 wd 예측
# column은 14개 뒤에서부터 144

#1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets.info())
print(datasets.shape)  # (420551, 14)

y_cor = datasets[-144:]['wd (deg)']  # 예측치 정답 데이터
print(y_cor.shape)

### 훈련할 데이터 자르기 ###
x_data = datasets[:-288].drop(['wd (deg)'], axis =1)
y_data = datasets[144:-144]['wd (deg)']

print(x_data.shape)  # (420263, 13)
print(y_data.shape)  # (420263,)


### 선생님이 작성한 split 코드 ###
# size_x = 144
# size_y = 144

# def split_x(dataset,size):
#     aaa = []
#     for i in range(len(dataset)- size + 1):
#         subset = dataset[i: (i+ size)]
#         aaa.append(subset)
#     return np.array(aaa)

# start_time = time.time()
# x = split_x(x_data, size_x)
# y = split_x(y_data, size_y)
# end_time = time.time()

# print(x.shape, y.shape)
# print("자르는시간:", round(end_time - start_time,2))

### 여기까지 선생님이 작성한 split 코드 ###

size = 144

def xy_split(x, y, size):
    x_split = []
    y_split = []
    for i in range(len(x_data) - size + 1):
        x_subset = x[i:i+size]
        y_subset = y[i]
        x_split.append(x_subset)
        y_split.append(y_subset)
    return np.array(x_split), np.array(y_split)


x_split, y_split = xy_split(x_data, y_data, size)

print(x_split.shape, y_split.shape)  # (420120, 144, 13) (420120,)

x_train, x_test, y_train, y_test = train_test_split(
    x_split, y_split,
    train_size = 0.7,
)


# scaling은 train 나눈후에 train만 이용해서
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)    # sklearn에서 fit은 보통 실행하라는 의미
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# submit용 x데이터
x_predict = datasets[-288:-144].drop(['wd (deg)'], axis =1)
print(x_predict.shape) # (288, 13)
x_predict = scaler.transform(x_predict)

# pandas dataframe reshape
x_predict = x_predict.values.reshape(-1,144,13)  
# # reshape 두번째 방법
# x_predict = x_predict.to_numpy()
# x_predict = x_predict.reshape(1,144,13)


#2. 모델구성

model= Sequential()
model.add(SimpleRNN(64, input_shape = (144,13)))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(144))

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

learning_rate = 0.01
model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 1000,
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
model.fit(x_split, y_split,
          epochs = 10,
        #   batch_size =    # keras batch size default = 32
        #   validation_split=0.2,  # shuffle 없이 뒤에서 부터 해당 비율을 잘라 validation으로 이용, validation_data와 validation_split 동시에 있을경우 validation_data가 override
          verbose =1,
        callbacks = [es, rlr])
end_time = time.time()

#4. 평가 예측


# y_test = np.array([[101], [102], [103], [104], [105], [106]])
# y_test = y_test.reshape(-1,1)

y_pred = model.predict(x_predict)


print("x_predict의 예측값: ", y_pred)
loss = model.evaluate(x_predict, y_cor)
print("loss: ", round(loss,2))

# x_predict의 예측값:  [[12.473992]]
print("걸린시간: ", round(end_time - start_time, 2))