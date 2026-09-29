# x_predict 변환할때 1차원 데이터를 2차원으로 reshape한뒤, split함수를 사용해서 3차원으로 변환
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))  

size = 6

# 데이터를 reshape한후, split_x 함수로 시계열데이터로 변환
# (N, 10, 1) -> (N, 5, 2)
# 결과를 뽑는다.


a = a.reshape(-1, 2)
print(a.shape)  # (50, 2)


def split_x(dataset, size):
    x = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        x.append(subset)
    return np.array(x)

bbb = split_x(a, size)

print(bbb.shape)  # (45, 6, 2)

x = bbb[:,:-1,:]   # x = [:, :-1] 
y = bbb[:,-1,1]    # y = [:,-1,-1]

print(x.shape, y.shape)  # (5, 4, 5, 2) (5, 2)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3, 1)))
# model.add(SimpleRNN(2, input_shape = (3, 1)))   # input_shape = (timesteps, input_dim=feature)
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# 파라미터 개수: units*features + units*bias + units*units
# timestpe이 3 -> hidden state도 3까지 존재
# model.add(LSTM(10, input_shape=(3,1)))
model.add(SimpleRNN(10, input_shape=(5,2)))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))

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
          epochs = 10,
          validation_split=0.2,
          verbose =1,
        callbacks = [es, rlr])


x_predict = x_predict.reshape(-1,5,2)

# y_test = np.array([[101], [102], [103], [104], [105], [106]])
# y_test = y_test.reshape(-1,1)

y_pred = model.predict(x_predict)


print("x_predict의 예측값: ", y_pred)
# loss = model.evaluate(x_predict, y_test)
# print("loss: ", round(loss,2))


### x_pred의 정답값(y_pred) 입력하는 부분 ###