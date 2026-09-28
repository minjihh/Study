# 시계열 데이터의 x데이터 시계열 모델의 input shape로 적합하게 잘라내기

import numpy as np

a = np.array(range(1, 11))  # 1~10
size = 5  # timestep 사이즈

print(a.shape) # (10,) 벡터

# def split_x(dataset, size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):  # range(6)
#         subset = dataset[i: (i + size)]  # i=0일때 dataset의 0~5번 인덱스값
#         aaa.append(subset)
#     return np.array(aaa)

# bbb = split_x(a, size)
# print(bbb)


def split_xy(dataset, size):
    x = []
    y = []
    for i in range(len(dataset) - size + 1):  # range(6)
        subset = dataset[i: (i + size)]  # i=0일때 dataset의 0~5번 인덱스값
        x.append(subset[:-1])
        y.append(subset[-1])
    return np.array(x), np.array(y)

x, y = split_xy(a, size)
print(x)
print(y)
print(x.shape) # (6, 4)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape, y.shape)  # (6, 4, 1) (6,)

print(x)



# exit()
# 모델링!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(10, input_shape = (4,1)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    verbose = 1,
    restore_best_weights= True
)

model.fit(x, y,
          epochs = 100,
          verbose= 1)




x_predict = np.array([7,8,9,10]).reshape(1, 4, 1)


y_predict = model.predict(x_predict)

print("[7,8,9,10]의 예측값: ", y_predict)
# 11의 예측값:  [[2.4614751]]
