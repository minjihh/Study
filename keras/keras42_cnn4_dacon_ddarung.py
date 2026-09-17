## test data reshape


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd
import time 


#1. 데이터

path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.info())

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

print(x.shape, y.shape)

# print(train_csv.columns())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler = scaler.fit(x)
x = scaler.transform(x)
test_csv = scaler.transform(test_csv)





print(x.shape, test_csv.shape)  # (929, 9) (200, 9)

x = x.reshape(-1,9,1,1)
test_csv = test_csv.reshape(-1,9,1,1)
x_test = x_test.reshape(-1,9,1,1)

# train_csv = train_csv.reshape(-1,9,1,1)
print(x.shape, test_csv.shape, x_test.shape) # 


#2. 모델 구성

# model = Sequential()

# model.add((Dense(3, input_dim = 9)))
# model.add(Dropout(0.5))

# model.add(Dense(1))

# input1 = Input(shape = (9,))
# dense1 = Dense(3)(input1)
# drop1 = Dropout(0.5)(dense1)
# output1 = Dense(1)(drop1)

# model = Model(inputs = input1, outputs = output1)

model = Sequential()
model.add(Conv2D(64, (3,1), input_shape=(9,1,1,), padding='same')) 
model.add(Conv2D(filters=32, kernel_size=(2,1), activation = 'relu', strides = 1, padding = 'same')) # (24,24,32)
model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,1), activation = 'relu', strides = 1, padding = 'same'))  # (23, 23, 32)
# model.add(Conv2D(16, (2, 2), activation = 'relu'))  # (22, 22, 16)
# model.add(Dropout(0.2))   # dropout의 파라미터의 개수 없음
# model.add(Conv2D(16, (2, 2), activation = 'relu'))  # (21, 21, 16)
# model.add(Dropout(0.2))   # dropout의 파라미터의 개수 없음
# model.add(Conv2D(16, (2, 2), activation = 'relu'))  # (20, 20, 16)

model.add(Flatten())
model.add(Dense(units=32, activation = 'relu'))  # units
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape = (32,), activation = 'relu'))
model.add(Dense(1))  


model.summary()

#3. 컴파일 훈련

model.compile(loss= 'mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True,
    verbose=1
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras34/'

filename = '{epoch:4d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k34_dropout_dacon_ddarung", date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose = 1,
    filepath=filepath 
)

start_time = time.time()
hist = model.fit(x, y, 
                 epochs=10, 
                 batch_size=4, 
                 verbose=1, 
                 validation_split=0.2, 
                #  callbacks = [es, mcp]
                 )
# model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1, validation_split=0.33)
end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(test_csv)


print("걸린시간: ", round(end_time - start_time, 2), "초")
# cpu 걸린시간:  2.73 초
# gpu 걸린시간:  4.72 초


# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))  

# rmse = RMSE(y_test, y_predict)
# print("RMSE :", rmse)

# submission['count'] = y_predict
# submission.to_csv(path + 'submit/submission_0911_MaxAbs.csv')




# #### plot 그리기 ####

# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))

# plt.plot(hist.history['loss'], c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], c = 'blue', label='val_loss')

# plt.title('ddarung loss')

# plt.xlabel('epoch')
# plt.ylabel('loss')

# plt.legend(loc = 'upper right')

# plt.grid()
# plt.show()