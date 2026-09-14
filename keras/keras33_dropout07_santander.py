# 이진분류 데이터 다중분류 방식으로 풀어보기
# 이진분류로 했을때랑 비교해보기
# 판다스의 series 데이터의 경우 vector형태라서 reshape이 안되는 경우가 있는데 이대 numpy로 변환해주면됨
# y = np.array(y) == y = y.to_numpy()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
import time
import datetime

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


x_train, x_val, y_train, y_val = train_test_split(x, y,
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
model.add(Dropout(0.5))

model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation = 'softmax'))


es = EarlyStopping(
    monitor='val_loss',
    mode ='auto',
    patience =100,
    restore_best_weights= True
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras31/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k30_dropout_santander", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=128, 
          validation_data=(x_val, y_val),
          callbacks = [es, mcp],
          verbose=1)
end_time = time.time()

loss = model.evaluate(x_val, y_val)
print("loss: ", loss[0])
print("acc: ", loss[1])

print("걸린시간: ", round((end_time-start_time),2), "초")

y_pred = model.predict(test_csv)
y_pred = np.argmax(y_pred, axis=1)  # argmax시 axis=1 설정 주의

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))  

# rmse = RMSE(y_test, y_pred)
# print("RMSE :", rmse)


submission_csv['target'] = y_pred
# submission_csv.to_csv(path + "submission_0911_MaxAbs.csv")






