# 20 카피
# california 데이터를 LSTM으로

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (20640, 8) (20640,)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)

# print(x_train.shape, y_train.shape)   # (14447, 8) (14447,)


### 시계열 데이터로 split ###


size = 5
def split_x(x, size):
    temp = []
    for i in range(len(x) - size + 1):  # range(6)
        subset = x[i: (i + size)]  # i=0일때 dataset의 0~5번 인덱스값
        temp.append(subset)
    return np.array(temp)

x = split_x(x, size)
print(x.shape)  # (20636, 5, 8)

y = split_x(y, size)
print(y.shape)  # (20636, 5)

###############################


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,)


#2. 모델구성
model = Sequential()
model.add(LSTM(5, input_shape=(5,8)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping  # EarlyStopping 카멜케이스

es = EarlyStopping(
    monitor='val_loss',
    mode = ' min',    # 사용하는 metric따라 다르게. accuracy라면 max가 좋음
    patience = 10,
    restore_best_weights=True,   # restore_best_weights 의 default 값은 False, 원칙은 True나 실제로 돌려보면 False일때 잘나오는 경우도 있음 (validation으로 판단하기 때문에 test data에서는 성능이 다를수도 있음)
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=1, 
                 batch_size=4, 
                 verbose=1, 
                 validation_data = (x_test, y_test),
                 callbacks = [es], # early stopping 적용하기
        #   valdiatoin_split=0.5
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")

print("===================== history ========================")
print(hist)
print("===================== hist.history ========================")
print(hist.history)
print("========================= loss ============================")
print(hist.history['loss'])    # hist.history 딕셔너리에 있는 loss값만 빼자.
print("======================== val_loss ===========================")
print(hist.history['val_loss'])    # hist.history 딕셔너리에 있는 val_loss값만 빼자.
print("============================================================")










