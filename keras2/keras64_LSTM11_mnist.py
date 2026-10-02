# 53-11 카피
# 28*28사이즈 이미지를 펼쳐서 28*28의 약수만큼의 타임스탬프를 사용한 LSTM으로 분류하는 작업
# split_x 함수부분 주의


# GAP: flatten 자리에 사용. 성능도 좋아지고 속도도 빨라짐# 파라미터 개수 계산하기: (필터개수 x (kenel_size(가로 x 세로) X input channel + 바이어스(커널당 하나)))
# convolution 적용시 입력채널이 여러개일 경우 채널마다 가중치가 달라지게 됨

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train,), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)  -> 흑백이미지의 경우 일반적으로 이렇게 많이 나타냄
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(y_train), np.max(y_train)) # 0 9

### 스케일링 1 MinMax
# mnist -> minmax scaling이 적절. maxabs도 동일
x_train = x_train/255.   # .을 붙임으로서 float 형태로 출력하게 됨
x_test = x_test/255.

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0


# 이미지 데이터 reshape
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)


# timestep = 7
# 함수 없이 바로 reshape
# x_train = x_train.reshape(-1, 7, 112)
# x_test = x_test.reshape(-1, 7, 112)

def split_x(dataset, timestep, feature_size=112):
    return dataset.reshape(-1, timestep, feature_size)


timestep = 7

x_train = split_x(x_train, timestep)
x_test = split_x(x_test, timestep)



print(x_train.shape)
# (60000, 7, 112)

print(x_test.shape)
# (10000, 7, 112)


# y_train, y_test는 변경하지 않는다.
print(y_train.shape)
# (60000,)

print(y_test.shape)
# (10000,)



#2. 모델구성

model = Sequential()
model.add(LSTM(64, input_shape=(7, 112),))  
model.add(Dense(10, activation = 'relu')) 

model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate),
              metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    restore_best_weights= True,
    verbose = 1,

)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
path = './_save/keras36/'
filename = '{epoch:4d}-{date:.4f}.keras'
filepath = "".join([path, "-", date, filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    verbose = 1
)


start_time = time.time()
model.fit(x_train, y_train, epochs = 1, batch_size=128,
          verbose = 1,
          validation_split=0.2,
          callbacks = [es, rlr])
end_time = time.time()


#4. 평가 예측
print("=============== model.evaluate ===============")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis = 1)  # 다중분류이므로 argmax 
# y_test = np.argmax(y_test, axis=1).reshape(-1,1)

print(y_predict.shape)  # (10000,)

acc_score = accuracy_score(
    y_test,
    y_predict
)


print("accuracy_score : ", acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')

# cpu 걸린시간: 558.48 초
# accuracy_score :  0.9901
# gpu 걸린시간 : 134.39 초
# accuracy_score :  0.989

# 0.995 맞추기 

## MaxPool 적용
# accuracy_score :  0.9897

## GAP 적용
# accuracy_score :  0.987
# 걸린시간 :  957.15 초

# lr 0.01
# accuracy_score :  0.9725
# 걸린시간 :  111.04 초

# lr 0.01, rlr
# accuracy_score :  0.974
# 걸린시간 :  92.74 초