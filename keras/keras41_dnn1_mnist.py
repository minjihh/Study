# 이미지 데이터를 DNN으로 분류하기

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
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

# #### 스케일링 2 MaxAbs
# x_train = (x_train - 127.5)/127.5   
# x_test = (x_test- 127.5)/127.5

# print(np.min(x_train), np.max(x_train)) # -1.0 1.0
# print(np.min(x_test), np.max(x_test)) # -1.0 1.0

x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)



from sklearn.preprocessing import OneHotEncoder   # sklearn의 OneHotEncoder 이용할때는 y값 reshape 후에 ohe 적용

ohe = OneHotEncoder(sparse_output=False)  # 혼동행렬 형태의 output False
y_train = y_train.reshape(-1,1)  
# y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델구성

model = Sequential()
model.add(Dense(64, input_shape = (784,)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))

# 실습 시작!
# 목표는 CNN을 이겨라

model.add(Dense(10, activation='softmax'))  # (10,)

model.summary()

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 100,
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
          callbacks = [es])
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


# dnn
# accuracy_score :  0.97

