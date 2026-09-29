import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))  # 101~106까지 찾자.

size = 6

# loss 0.1 이하
# 결과는 [101, 102, 103, 104, 105, 106]의 근사치가 나오면 됨

def xy_split(a,size):
    x = []
    y = []
    for i in range(len(a) - size + 1):
        subset = a[i:i + size]
        # print(subset)
        x.append(subset[:-1])
        y.append(subset[-1])
    return np.array(x), np.array(y)

pred_size = 5
def x_pred_split(x_predict, pred_size):
    x_pred = []
    for i in range(len(x_predict) - pred_size + 1):
        subset = x_predict[i : i + pred_size]
        x_pred.append(subset[:])
    return np.array(x_pred)


x, y = xy_split(a, size)
x_pred = x_pred_split(x_predict, pred_size)

print(x.shape, y.shape)  # (95, 5) (95,)


print(x_pred.shape)   # (6, 5)

x = x.reshape(95,5,1)
x_pred = x_pred.reshape(6,5,1)
# print(y[:3])


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3, 1)))
# model.add(SimpleRNN(2, input_shape = (3, 1)))   # input_shape = (timesteps, input_dim=feature)
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# 파라미터 개수: units*features + units*bias + units*units
# timestpe이 3 -> hidden state도 3까지 존재
# model.add(LSTM(10, input_shape=(3,1)))
model.add(SimpleRNN(64, input_shape=(5,1), activation='linear'))  # activatoin을 linear로 설정하면 tanh대신에 linear적용
model.add(Dense(16, activation = 'linear'))
model.add(Dense(8, activation = 'linear'))

model.add(Dense(1))

# model.summary()

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
          epochs = 1500,
          validation_split=0.2,
          verbose =1,
        callbacks = [es, rlr])

y_test = np.array([[101], [102], [103], [104], [105], [106]])
y_test = y_test.reshape(-1,1)
# print(y_test.shape)

y_pred = model.predict(x_pred)

# print(y_pred.shape)  # (6, 1)

print("x_predict의 예측값: ", y_pred)
loss = model.evaluate(x_pred, y_test)
print("loss: ", round(loss,2))
