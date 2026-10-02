# GAP: flatten 자리에 사용. 성능도 좋아지고 속도도 빨라짐# 파라미터 개수 계산하기: (필터개수 x (kenel_size(가로 x 세로) X input channel + 바이어스(커널당 하나)))
# convolution 적용시 입력채널이 여러개일 경우 채널마다 가중치가 달라지게 됨

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
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

print(y_train.shape, y_test.shape)  # (60000,) (10000,)



from tensorflow.keras.layers import Reshape

print(x_train.shape)  # (60000, 28, 28)

#2. 모델구성
# Reshape를 제일 첫번째에 놔도 됨

model = Sequential()
model.add(Reshape(target_shape = (28, 28, 1), input_shape= (28, 28)))  # reshape -> 순서, 값이 바뀌면 안됨
# (N, 28, 28, 1)으로 바뀜
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))  # (26,26,64) # conv2D에 넣기위한 4차원 데이터로 mnist 데이터의 shape 변환필요 # 첫번재 층에서는 linear
model.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu')) # (24,24,32)
model.add(Conv2D(16,(2,2), activation = 'relu'))  # (23, 23 , 16)
####################################################
model.add(Reshape(target_shape = (23*23, 16)))
model.add(LSTM(10,))  # (N, 23*23, 10)
####################################################
# model.add(Flatten())
# model.add(GlobalAveragePooling2D())   # 4차원을 입력으로 받음 (N, H, W, C)
model.add(Dense(units=32, activation = 'relu'))  # units
model.add(Dense(10, activation='softmax'))  # (10,)

model.summary()

exit()

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', 
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
model.fit(x_train, y_train, epochs = 100000, batch_size=128,
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
y_predict = np.argmax(y_predict, axis = 1).reshape(-1,1)  # 다중분류이므로 argmax # reshape 안붙여도됨
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score =accuracy_score(y_test, y_predict)
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