# 아모레, 하이닉스의 5개 컬럼 이상 사용해서 삼성전자의 내일 시가 맞추기
# 앙상블쓰기
# 아모레, 하이닉스 모델에 LSTM과 Conv1D 이용


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, Dense, Conv1D, Flatten, Input, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import time


#1. 데이터
path = './_data/앙상블용/'

datasets1 = pd.read_csv(path + '아모레261012.csv', index_col=0, thousands =',', encoding="cp949")
datasets2 = pd.read_csv(path + '하이닉스261012.csv', index_col=0, thousands =',', encoding="cp949")
datasets3 = pd.read_csv(path + '삼성전자261012.csv', index_col=0, thousands =',', encoding="cp949")

# datasets1 = datasets1.sort_values(by="일자", ascending = True)
# datasets2 = datasets2.sort_values(by="일자", ascending = True)
# datasets3 = datasets3.sort_values(by="일자", ascending = True)

# 기록된 데이터가 가장 짧은 삼성전자 기준으로 데이터 자르기
# 날짜를 datetime으로 변경하고 날짜 기준 오름차순 정렬
for datasets in [datasets1, datasets2, datasets3]:
    datasets.index = pd.to_datetime(datasets.index, format='mixed')
    datasets.sort_index(inplace=True)

datasets1 = datasets1.loc['2022-03-14':'2026-10-12']
datasets2 = datasets2.loc['2022-03-14':'2026-10-12']
datasets3 = datasets3.loc['2022-03-14':'2026-10-12']

print(datasets1.info())
print(datasets1.shape)  # (1120, 16)

print(datasets2.info())
print(datasets2.shape)  # (1120, 16)

print(datasets3.info())
print(datasets3.shape)  # (1120, 16)


y_cor = datasets3[-1:]['시가'].to_numpy(dtype= np.float32)
y_cor = y_cor.reshape(1, -1)

# y_cor = datasets[-144:]['wd (deg)']  # 예측치 정답 데이터
# print(y_cor.shape)  

# y_cor = y_cor.values.reshape(1, -1)




### 훈련할 데이터 자르기 ###
# x_data = datasets[:-288].drop(['wd (deg)'], axis =1).to_numpy(dtype= np.float32)
# y_data = datasets[144:-144]['wd (deg)'].to_numpy(dtype=np.float32)
# timestamp = 5

x1_data = datasets1[:-10][['시가', '고가', '저가', '종가', '거래량']].to_numpy(dtype= np.float32)
print(x1_data.shape)  # (1110, 5)



x2_data = datasets2[:-10][['시가', '고가', '저가', '종가', '거래량']].to_numpy(dtype= np.float32)
print(x2_data.shape)  # (1110, 5)

y_data = datasets3[5:-5]['시가'].to_numpy(dtype=np.float32)
print(y_data.shape)  # (1110,)



### split ###

size = 5

def xy_split(x, y, size):
    x_split = []
    y_split = []
    for i in range(len(x) - size + 1):
        x_subset = x[i:i + size]
        y_subset = y[i:i + size]
        x_split.append(x_subset)
        y_split.append(y_subset)
    return np.array(x_split), np.array(y_split)


split_start_time = time.time()
x1_split, y_split = xy_split(x1_data, y_data, size)
x2_split, y_split = xy_split(x2_data, y_data, size)

split_end_time = time.time()
print("자르는시간:", round(split_end_time - split_start_time,2))


print(x1_split.shape, x2_split.shape, y_split.shape)  #  (1106, 5, 5) (1106, 5, 5) (1106, 5)



np_path = './_data/stock_npy/'
np.save(np_path + 'keras70_stock_x1_split.npy', arr=x1_split)
np.save(np_path + 'keras70_stock_x2_split.npy', arr=x2_split)
np.save(np_path + 'keras70_stock_y_split.npy', arr=y_split)



# x_split = np.load(np_path + 'keras58_jena_x.npy')
# y_split = np.load(np_path + 'keras58_jena_y.npy')


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_split, x2_split, y_split,
    train_size = 0.7,)


# scaling 하기전에 2차원 데이터로 reshape 필요
x1_train_scaler = x1_train.reshape(-1, x1_train.shape[-1])
x1_test_scaler = x1_test.reshape(-1, x1_test.shape[-1])

x2_train_scaler = x2_train.reshape(-1, x2_train.shape[-1])
x2_test_scaler = x2_test.reshape(-1, x2_test.shape[-1])




# scaling은 train 나눈후에 train만 이용해서
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train_scaler)    # sklearn에서 fit은 보통 실행하라는 의미
x1_train = scaler.transform(x1_train_scaler)
x1_test = scaler.transform(x1_test_scaler)

x1_train = x1_train.reshape(-1,5,5)
x1_test = x1_test.reshape(-1,5,5)

scaler.fit(x2_train_scaler)    # sklearn에서 fit은 보통 실행하라는 의미
x2_train = scaler.transform(x2_train_scaler)
x2_test = scaler.transform(x2_test_scaler)

x2_train = x2_train.reshape(-1,5,5)
x2_test = x2_test.reshape(-1,5,5)


print(x1_train.shape, x2_train.shape, y_train.shape)  #  (774, 5, 5) (774, 5, 5) (774, 5)



# submit용 x데이터
# x_predict = datasets[-288:-144].drop(['wd (deg)'], axis =1)
# x_predict = datasets[-288:-144].drop(['T (degC)'], axis =1)

x1_predict = datasets1[-5:][['시가', '고가', '저가', '종가', '거래량']].to_numpy(dtype= np.float32)
x2_predict = datasets2[-5:][['시가', '고가', '저가', '종가', '거래량']].to_numpy(dtype= np.float32)


print(x1_predict.shape, x2_predict.shape) # (5, 5) (5, 5)



# x_predict scaling

scaler.fit(x1_train_scaler) 
x1_predict = scaler.transform(x1_predict)
x1_predict = x1_predict.reshape(-1,5,5)
print(x1_predict.shape)  # (222, 5, 5)


scaler.fit(x2_train_scaler) 
x2_predict = scaler.transform(x2_predict)
x2_predict = x2_predict.reshape(-1,5,5)
print(x2_predict.shape)  # (222, 5, 5)

# pandas dataframe reshape
# x_predict = x_predict.values.reshape(-1,144,13)  
# # reshape 두번째 방법
# x_predict = x_predict.to_numpy()





#2. 모델구성

model= Sequential()

#2-1. 모델(함수형)
input1 = Input(shape = (5,5))
dense1 = Conv1D(10, kernel_size = 5, activation = 'relu', name = 'han1')(input1)
dense2 = Dense(20, activation = 'relu', name = 'han2')(dense1)
dense3 = Dense(30, activation = 'relu', name = 'han3')(dense2)
output1 = Dense(1, activation = 'relu', name = 'han4')(dense3)
output1 = Flatten()(output1) 

#2-2. 모델(함수형)
input21 = Input(shape = (5,5))
# dense21 = Conv1D(10, kernel_size = 5, activation = 'relu', name = 'han21')(input21)
dense21 = LSTM(10)(input21)
dense22 = Dense(40, name = 'han22')(dense21)
dense23 = Dense(30, name = 'han23')(dense22)
dense24 = Dense(20, name = 'han24')(dense23)
output21 = Dense(1, name = 'han25')(dense24)

#2-3. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
# 대문자 Concatenate는 클래스, 소문자는 method 또는 함수
# 대문자 concat, 소문자 concat 기능은 같고 사용 방법이 다름

# merge1 = concatenate([output1, output21], name = 'mg1')
merge1 = Concatenate(name = 'mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input21], outputs = last_output)
model.summary()


#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### mcp 파일명 ###
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras70/'
filename = '{epoch:04d}-{val_loss:.4f}_stock.keras'
filepath = "".join([pt_path, "k70_", date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    save_best_only=True,
    verbose=1
)

learning_rate = 0.01
model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
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


start_time = time.time()
model.fit([x1_train, x2_train], y_train,
          epochs = 1000,
          batch_size = 600,    # keras batch size default = 32
        #   validation_split=0.2,  # shuffle 없이 뒤에서 부터 해당 비율을 잘라 validation으로 이용, validation_data와 validation_split 동시에 있을경우 validation_data가 override
          validation_data = ([x1_test, x2_test], y_test),
          verbose =1,
        callbacks = [es, rlr, mcp])
end_time = time.time()

#4. 평가 예측


y_pred = model.predict([x1_predict, x2_predict])
y_pred = y_pred.reshape(-1,1)

print("걸린시간: ", round(end_time - start_time, 2))

print("x_predict의 예측값: ", y_pred)
loss = model.evaluate([x1_test, x2_test], y_test)
print("loss: ", round(loss, 2))


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_cor, y_pred)
print("RMSE :", rmse)

# k70_1012_1348-0154-326637792.0000_stock
# 258371.39

# k70_1012_1503-0184-237756720.0000_stock
# 258405.28

# k70_1012_1506-0093-282049632.0000_stock
# 254457.9

# k70_1012_1525-0134-260450336.0000_stock
# 257806.27

# k70_1012_1536-0210-198750064.0000_stock
# 261017.11


## Conv1D + LSTM
# k70_1012_1547-0136-117693504.0000_stock
# 264045.5

# k70_1012_1549-0156-127654544.0000_stock
# 260647.92