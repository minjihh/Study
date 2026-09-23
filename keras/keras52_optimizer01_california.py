# 27-1 카피
# 스케일링 -> 모든 x값에 대해서 같은 비율로 처리하면 데이터 조작 문제 없음
# x의 값이 너무 클 경우, 역전파할때도 스케일링 하는게 성능향상에 도움될 수 있음
# 역전파시 가장 잘되는 데이터 -> 부동소수점 형태
# x값은 0~1사이 값으로 바꾸고 싶을때 -> (x-min)/(max-min) (=minmax scaler)
# 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)    # sklearn에서 fit은 보통 실행하라는 의미
x = scaler.transform(x)

print(x)
print(np.min(x), np.max(x))
# 0.0 1.0000000000000002


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)

print(x_train.shape, y_train.shape)   # (14447, 8) (14447,)



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1, validation_data = (x_val, y_val)
        #   valdiatoin_split=0.5
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")



def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# 0.7308131612410732

# lr: 0.01
# 걸린시간:  92.08 초
# RMSE : 0.7272736989511669





