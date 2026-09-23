from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 이때 3단계에서 validation_data하면 어떤 데이터가 va/test로 쪼개지는지?

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model =Sequential()
model.add((Dense(3, input_dim=13)))
model.add(Dense(1))


#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data= (x_val, y_val))
# model.fit(x_train, y_train, epochs=10, batch_size=4, validation_split= 0.33)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# x 전체로 scaling : # RMSE : 8.672873642737752
# x_train만으로 MinMax scaling : RMSE : 12.013473382237372
# x_train만 가지고 Standard스케일링: 20.39037099368951
# x_train만 가지고 MaxAbs 스케일링: 11.12472809697239
# x_train만 가지고 Robust 스케일링: 
# lr: 0.01 -> 4.055655831771788
