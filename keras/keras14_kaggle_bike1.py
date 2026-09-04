# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "./_data/kaggle_bike/" 

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)  # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
print(submission)   # [6493 rows x 1 columns]

print(train_csv.shape)    # (10886, 11)
print(test_csv.shape)     # (6493, 8)
print(submission.shape)   # (6493, 1)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())


############ 결측치 확인 ##############

print(train_csv.isna().sum())
print(test_csv.isna().sum())  # 동일: print(test_csv.isnull().sum())



############ x, y 분리 ##############
x = train_csv.drop(['casual','registerd', 'count'], axis=1)   # 두개 이상은 list
print(x)




