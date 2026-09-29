# 데이터 56-2 카피
# 다대다 RNN
# LSTM을 다층쌓는다고해서 다 좋은건 아님 (LSTM 한번 통과한 데이터 -> 시계열 데이터라고 볼수 있을지 자체가 의문)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10, 11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]
            ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])   # 80 맞춰보아요.


print(x.shape, y.shape)  # (13, 3) (13,)

x=x.reshape(-1,3,1)
print(x.shape)


#2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3,1), return_sequences = True, ))
model.add(LSTM(5, return_sequences=True, activation='linear'))
model.add(LSTM(5, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1))

model.summary()


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

model.fit(x, y,
          epochs = 1000,
          validation_split=0.2,
          verbose =1,
        callbacks = [es, rlr])

#4. 평가 예측


x_predict = x_predict.reshape(-1,3,1)

# y_test = np.array([[101], [102], [103], [104], [105], [106]])
# y_test = y_test.reshape(-1,1)

y_pred = model.predict(x_predict)


print("x_predict의 예측값: ", y_pred)
# loss = model.evaluate(x_predict, y_test)
# print("loss: ", round(loss,2))
# x_predict의 예측값:  [[12.473992]]