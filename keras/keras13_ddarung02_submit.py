# 데이터 -> 이상치, 결측치 처리후 사용
# 데이콘 설명 링크: https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd


#1. 데이터

path = "./_data/ddarung/"   # 상대경로
# path = "c:/study/_data/ddarung/"    # 절대경로 #  /, \ 섞어서 써도 상관없으나 "\n" 주의
# path = "c:\study\_data\ddarung\"    # "\s"를 인식해서 에러
# path = "c://study//_data//ddarung/"    # // 로 구별해도 상관없음. /로 문제 있을 경우 //사용, fsspec 다운받으러는 에러 발생해서 다운완료 상태.
# path = "c:\\study\\_data\\ddarung\\"    # \\ 로 구별해도 상관없음. \로 문제 있을 경우 //사용, "\t"의 경우 탭으로 인식할 가능성 주의
# path = "c:\\study\_data\\ddarung/"    #  /, \ 섞어서 사용할 경우 코드실행은 문제없으나 가급적 비권장




train_csv = pd.read_csv(path + "train.csv", index_col=0)  # index_col 표기해서 데이터로 쓰지 않도록 함, index_col=0 을 써서 default로 들어가는 첫번째 index column 삭제시킴 // 모델 훈련전 인덱스 포함시키지 않도록 전처리
#첫컬럼에 index 없는 경우는 나중에 설명

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

print(train_csv.columns)   # ".columns -> 판다스 문법"
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')

print(train_csv.info())   # ".info() -> 판다스 문법"
print(test_csv.info())


################################### 결측치 처리 1. 삭제 ###############################################

train_csv = train_csv.dropna()   # 결측치를 제거하겠다, 결측치가 있는 행 모두 삭제 
print(train_csv) # [1328 rows x 10 columns]

##################### train_csv를 x와 y로 분리  #####################
x = train_csv.drop(['count'], axis=1)  # count 칼럼을 삭제하겠다.   // axis = 0 -> 행 , axis = 1 -> 열
# train_csv 데이터는 건들지 않은채로 필요한 부분만 x로 지정

print(x)  # [1328 rows x 9 columns]

y = train_csv['count']  # 컬럼하나 -> 벡터형태와 동일
print(y)
print(y.shape)  # (1328,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=50)

############## submit 물밑작업 ##############

print(test_csv.info())

################################### 결측치 처리 2. 평균값 넣기 ###############################################







exit()

#2. 모델 구성

model = Sequential()
model.add(Dense(5, input_dim= 9))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=2)


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
y_predict = model.predict(x_test)

rmse = root_mean_squared_error(y_test, y_predict)
print("rmse: ", rmse)


############################################################

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))







