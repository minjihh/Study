import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10, 11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]
            ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])   # 80 맞춰보아요.

# [실습] 맹그러봐!!

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (13, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3, 1)))
# model.add(SimpleRNN(2, input_shape = (3, 1)))   # input_shape = (timesteps, input_dim=feature)
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# 파라미터 개수: units*features + units*bias + units*units
# timestpe이 3 -> hidden state도 3까지 존재
model.add(LSTM(10, input_shape=(3,1)))
# model.add(GRU(10, input_shape=(3,1)))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer= 'adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    verbose = 1,
)

# RNN batch_size: 자동으로 계산되기 때문에 input_shape만 잘 입력해주면 됨
model.fit(x, y, 
          epochs = 3000,
          callbacks = [es])


#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_predict = x_predict.reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 :', y_predict)

# GRU [50,60,70]의 결과 : [[75.24482]]
# LSTM [50,60,70]의 결과 : [[75.31073]]