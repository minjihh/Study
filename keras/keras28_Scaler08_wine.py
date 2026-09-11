from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
import numpy as np

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48]))


y = pd.get_dummies(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(3, activation='softmax'))



#3. 컴파일, 훈련

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience =550,
    restore_best_weights=True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100000, batch_size=16, validation_split=0.2,  callbacks = [es]) 
end_time = time.time()

# 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss[0])
print("acc: ", round(loss[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score

accuracy_score = accuracy_score(y_test,y_pred)
print('acc_score: ', accuracy_score)

print("걸린시간: ", round(end_time - start_time,2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# x 전체가지고 스케일링: 0.8050764858994133
# x_train만 가지고 MinMax 스케일링: 0.4714045207910317
# x_train만 가지고 Standard 스케일링: 0.23570226039551584
# x_train만 가지고 MaxAbs 스케일링: 0.3333333333333333
# x_train만 가지고 Robust 스케일링: 

# acc = 0.95