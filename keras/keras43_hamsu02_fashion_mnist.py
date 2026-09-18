from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

# 실습 맹그러봐요

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(np.unique(y_train, return_counts=True))



### 스케일링 1 MinMax
# mnist -> minmax scaling이 적절. maxabs도 동일
x_train = x_train/255.   # .을 붙임으로서 float 형태로 출력하게 됨
x_test = x_test/255.

print(np.min(x_train), np.max(x_train)) # 0.0 1.0
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape)  # 


# one hot encoding
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)



#2. 모델구성

# model = Sequential()
# model.add(Dense(64, input_shape=(784,)))
# model.add(Dense(units = 32, activation = 'relu'))
# model.add(Dense(10))
# model.add(Dense(units = 16, activation = 'relu'))
# model.add(Dense(10, activation = 'softmax'))


input1 = Input(shape=(784,))
dense1 = Dense(64)(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(16, activation='relu')(dense3)
output1 = Dense(10, activation = 'softmax')(dense4)

model = Model(inputs = input1, outputs = output1)

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 250,
    restore_best_weights= True,
    verbose = 1
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 100000,
          batch_size = 128,
          verbose = 1,
          validation_split=0.2,
        #   metrics = ['acc'],
          callbacks = [es]
          )
end_time = time.time()




loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis =1)

acc_score = accuracy_score(y_pred, y_test) 

print("acc_score: ", acc_score)

print("걸린시간: ", round(end_time - start_time, 2), '초')


import matplotlib.pyplot as plt

# plt.imshow(x_train[50],) # 'gray')  # 원래 grayscale 이미지에서 'gray' 설정 안했다고해서 color 이미지라고 할 순 없음
# plt.show()

# maxpool 적용
# acc_score:  0.9034

# dnn을 gpu에서 실행
# acc_score:  0.8492
# 걸린시간:  649.75 초