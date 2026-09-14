from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


path = './_save/keras31/'


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
# model =Sequential()
# model.add((Dense(3, input_dim=13)))
# model.add(Dense(1))

model = load_model(path + 'k30_0914_1353-0001-50667.1953.keras')


#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=10,
    restore_best_weights=True,
    verbose = 1
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k30_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = filepath,
    save_best_only=True,
    verbose = 1

)

model.compile(loss='mse', optimizer='adam')
# hist = model.fit(x_train, y_train, 
#                 epochs=10, 
#                 batch_size=4, 
#                 validation_data= (x_val, y_val),
#                 callbacks=[es, mcp])


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print("mse: ", loss)

r2 = r2_score(y_test, results)
print("r2: ", r2)

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


# mse:  611.0635375976562
# r2:  -5.868082091195529
# RMSE : 24.719699338997366