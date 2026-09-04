# https://www.kaggle.com/competitions/bike-sharing-demand/data
# activation 함수 -> 레이어마다 적용
# relu 사용해서 역전파시 속도도 빠르고 성능도 좋아짐.

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
x = train_csv.drop(['casual','registered', 'count'], axis=1)   # 두개 이상은 list
print(x)   # [10886 rows x 8 columns]

y = train_csv['count']
print(y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7, random_state=56)


### 모델 구성 ###

model = Sequential()
model.add((Dense(5, activation='relu', input_dim = 8)))  # relu: 값들을 양수만 나오도록
model.add(Dense(9, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1)) # activation없을때, default linear

### 컴파일, 훈련 ###

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=16)

### 평가, 추론 ###

loss = model.evaluate(x_test,y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE: ', rmse)

y_submit = model.predict(test_csv)


submission['count'] = y_submit
# print(submission)
# print(submission.shape)

submission.to_csv(path + "submit/submission_0904_0504.csv")
