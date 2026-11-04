# 맹글

import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
 
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.7, random_state=1)
print(x_train.shape, y_train.shape)  # (7620, 8) (7620,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# k-fold

n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 333)

#2. 모델
model = GradientBoostingRegressor()


#3,4 훈련(컴파일 없음), 평가
scores= cross_val_score(model, x_train, y_train, cv = kfold)  # cross_val_score가 model.fit까지 시킴

print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

print(y_predict)
print(y_test)

# 맹그러바 

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))  

# rmse = RMSE(y_test, y_predict)
# print("RMSE :", rmse)


from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

# GradientBoostingRegressor
# RMSE : 150.98401774482068