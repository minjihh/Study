import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
            [9,8,7,6,5,4,3,2,1,0],
             ]).T   

print(a.shape)  # (10, 2)

# data 자르기
size = 5  # timestep + 1




 
######### split 함수 안에서 자르기  ######### 

# def split_x(data, size):
#     x = []
#     y = []
#     y_use = []
#     for i in range(len(data) - size + 1):
#         subset = data[i:i+size]
#         x.append(subset[:-1])
#         y.append(subset[-1])
#         y_use.append(y[i][-1])
#     return np.array(x), np.array(y), np.array(y_use)

# x, y, y_use = split_x(a, size)


# # x, y= split_x(a, size)
# print(y)
# print(y_use)

# exit()

# print(y_use.shape)
# # y_use = y_use.reshape(7,1)
# # print(y_use.shape)
# print(x.shape, y_use.shape)  #  (7, 3, 2) (7, 1)



########################

#### 기존방식 ####


def split_x(data, size):
    x = []
    for i in range(len(data) - size + 1):
        subset = data[i:i+size]
        x.append(subset[:-1])
    return np.array(x)

bbb = split_x(a, size)

print(bbb.shape)  # (6, 4, 2)

# exit()

x = bbb[:,:-1,:]   # x = [:, :-1] 
y = bbb[:,-1,1]    # y = [:,-1,-1]


print(x.shape, y.shape)  # (6, 3, 2) (6,)

# exit()




# 모델링!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(5, input_shape = (3,2)))
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



x_predict = np.array([[8,2], [9,1], [10, 0]]).reshape(1,3,2)

y_predict = model.predict(x_predict)

print("[[8,2], [9,1], [10, 0]]의 에측값: ", y_predict)
# [[8,2], [9,1], [10, 0]]의 에측값:  [[1.1942272]]
# [[7,3], [8,2], [9,1], [10, 0]]의 에측값:  [[2.4428406]]
# [[8,2], [9,1], [10, 0]]의 에측값:  [[1.7039216]]