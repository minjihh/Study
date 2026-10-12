# 아모레, 하이닉스의 5개 컬럼 이상 사용해서 삼성전자의 내일 시가 맞추기
# 앙상블쓰기
# 아모레, 하이닉스 모델에 LSTM과 Conv1D 이용


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Conv1D, Flatten, Input, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import time


#1. 데이터
path = './_data/앙상블용/'

datasets1 = pd.read_csv(path + '아모레261012.csv', index_col=0, thousands =',', encoding="cp949")
datasets2 = pd.read_csv(path + '하이닉스261012.csv', index_col=0, thousands =',', encoding="cp949")
datasets3 = pd.read_csv(path + '삼성전자261012.csv', index_col=0, thousands =',', encoding="cp949")


# 기록된 데이터가 가장 짧은 삼성전자 기준으로 데이터 자르기
# 날짜를 datetime으로 변경하고 날짜 기준 오름차순 정렬
for datasets in [datasets1, datasets2, datasets3]:
    datasets.index = pd.to_datetime(datasets.index, format='mixed')
    datasets.sort_index(inplace=True)

datasets1 = datasets1.loc['2022-03-14':'2026-10-12']
datasets2 = datasets2.loc['2022-03-14':'2026-10-12']
datasets3 = datasets3.loc['2022-03-14':'2026-10-12']


y_cor = datasets3[-1:]['시가'].to_numpy(dtype= np.float32)
y_cor = y_cor.reshape(1, -1)


# ### 훈련할 데이터 자르기 ###
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


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_split, x2_split, y_split,
    train_size = 0.7,)


# scaling 하기전에 2차원 데이터로 reshape 필요
x1_train_scaler = x1_train.reshape(-1, x1_train.shape[-1])
x1_test_scaler = x1_test.reshape(-1, x1_test.shape[-1])

x2_train_scaler = x2_train.reshape(-1, x2_train.shape[-1])
x2_test_scaler = x2_test.reshape(-1, x2_test.shape[-1])



# # scaling은 train 나눈후에 train만 이용해서
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x1_predict = datasets1[-5:][['시가', '고가', '저가', '종가', '거래량']].to_numpy(dtype= np.float32)
x2_predict = datasets2[-5:][['시가', '고가', '저가', '종가', '거래량']].to_numpy(dtype= np.float32)


print(x1_predict.shape, x2_predict.shape) # 



# x_predict scaling

scaler.fit(x1_train_scaler) 
x1_predict = scaler.transform(x1_predict)
x1_predict = x1_predict.reshape(-1,5,5)
print(x1_predict.shape)  # (222, 5, 5)


scaler.fit(x2_train_scaler) 
x2_predict = scaler.transform(x2_predict)
x2_predict = x2_predict.reshape(-1,5,5)
print(x2_predict.shape)  # (222, 5, 5)



pt_path = './_save/keras70/'

model = load_model(pt_path + 'k70_1012_1547-0136-117693504.0000_stock.keras')

model.summary()



#4. 평가 예측
y_pred = model.predict([x1_predict, x2_predict])
y_pred = y_pred.reshape(-1,1)


print("x_predict의 예측값: ", y_pred)
loss = model.evaluate([x1_predict, x2_predict], y_cor)
print("loss: ", round(loss, 2))


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_cor, y_pred)
print("RMSE :", rmse)