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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
from tensorflow.keras.optimizers import Adam 
# learning_rate = 0.01
# learning_rate = 0.001  # keras default
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009



path_pt = 'c:/Users/Admin/Desktop/Study_data/_save/keras53/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}-california.keras'

file_path = "".join([path_pt, date,filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = file_path,
    save_best_only=True,
    verbose =1

)
es = EarlyStopping(
    monitor ='val_loss',
    mode = 'auto',
    verbose = 1,
    patience=40,
    restore_best_weights = True,
)


model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))


rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=500, 
                batch_size=32, 
                verbose=1,
                validation_split = 0.2,
                callbacks = [es, rlr],
        #   valdiatoin_split=0.5
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")
print("mse: ", loss)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# 0.7308131612410732

# lr: 0.01
# 걸린시간:  92.08 초
# RMSE : 0.7272736989511669

# rlr 적용
# 걸린시간:  25.19 초
# mse:  0.5310698747634888
# RMSE : 0.7287453498461379

# lr: 0.0001, rlr 적용
# 걸린시간:  76.71 초
# mse:  0.5328271985054016
# RMSE : 0.7299501065443597