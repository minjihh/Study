# 데이터 -> 이상치, 결측치 처리후 사용
# 데이콘 설명 링크: https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


#1. 데이터

path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)  # index_col 표기해서 데이터로 쓰지 않도록 함. // 모델 훈련전 인덱스 포함시키지 않도록 전처리
print(train_csv)  # id열 포함 [1459 rows x 11 columns]  // id열 안포함 [1459 rows x 10 columns]
# 출력시 제일 첫 컬럼은 실제 데이터에서는 존재하지 않는 컬럼이지만 판다스에서 보여줌
# 판다스에서 통상적으로 첫번째 row는 컬럼명으로 데이터에 포함시키지 않음
# train csv내 train/test분리하기 




test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)  # [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)

print(train_csv.shape)  # (1459, 10)
print(test_csv.shape)   # (715, 9)
print(submission.shape)  # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')

print(train_csv.info())
print(test_csv.info())

exit()
################################### 결측치 처리 1. 삭제 ###############################################

train_csv = train_csv.dropna()
print(train_csv) # [1328 rows x 10 columns]








