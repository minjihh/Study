# 이진분류 데이터 다중분류 방식으로 풀어보기
# 이진분류로 했을때랑 비교해보기
# 판다스의 series 데이터의 경우 vector형태라서 reshape이 안되는 경우가 있는데 이대 numpy로 변환해주면됨
# y = np.array(y) == y = y.to_numpy()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
import time

path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.columns)


x = train_csv.drop(['target'], axis = 1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)
print(np.unique(y, return_counts=True))  # (array([0, 1]), array([179902,  20098]))

y = pd.get_dummies(y, dtype=int)
print(y)
print(type(y))  # <class 'pandas.DataFrame'>


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                  train_size=0.8,
                                                  random_state=1,
                                                  stratify=y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(5, input_dim=200, activation='relu' ))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


es = EarlyStopping(
    monitor='val_loss',
    mode ='auto',
    patience =10,
    restore_best_weights= True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=128, 
          validation_data=(x_test, y_test),
          callbacks = [es, rlr])
end_time = time.time()

loss = model.evaluate(x_test, y_test)
print("loss: ", loss[0])
print("acc: ", loss[1])

print("걸린시간: ", round((end_time-start_time),2), "초")

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)  # argmax시 axis=1 설정 주의
y_test = np.argmax(y_test, axis = 1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)


# submission_csv['target'] = y_pred
# submission_csv.to_csv(path + "submission_0911_MaxAbs.csv")

# lr 0.01 일때, patience 10 -> 
# RMSE : 0.2935557868617139




