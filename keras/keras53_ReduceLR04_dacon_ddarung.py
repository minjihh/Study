import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd
import time 


#1. 데이터

path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.info())

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

print(x.shape, y.shape)

# print(train_csv.columns())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)

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

model = Sequential()
model.add((Dense(3, input_dim = 9)))
model.add(Dense(1))

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009


model.compile(loss= 'mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, 
                 batch_size=4, 
                 verbose=1, 
                 validation_data=(x_val, y_val), 
                 callbacks = [es, rlr])
# model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1, validation_split=0.33)
end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print("걸린시간: ", round(end_time - start_time, 2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# submission['count'] = y_predict
# submission.to_csv(path + 'submit/submission_0911_MaxAbs.csv')



# lr: 0.01 -> 49.76008647285095
# lr: 0.01, rlr -> 49.48989955465273