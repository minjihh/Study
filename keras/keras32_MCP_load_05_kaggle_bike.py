# 모든 loss들은 음수값을 방지하기 위한 장치가 들어간다.

import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터
 
path = './_data/kaggle_bike/'
pt_path = './_save/keras31/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.7, random_state=1)
print(x_train.shape, y_train.shape)  # (7620, 8) (7620,)

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

# model = Sequential()
# model.add((Dense(5, activation='relu', input_dim = 8)))  # relu: 값들을 양수만 나오도록
# model.add(Dense(9, activation='relu'))
# model.add(Dense(13, activation='relu'))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1)) # activation없을때, default linear // 통상적으로 마지막 layer에는 activation 넣지 않음

model = load_model(pt_path + 'k30_0914_1419-0034-22516.6543.keras')


#3. 컴파일 훈련
### mse  ###
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True,
    verbose = 1
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k30_", date, "-", filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)

# start_time =time.time()
# hist = model.fit(x_train, y_train, 
#                  epochs = 1000, 
#                  batch_size=4, 
#                  validation_split=0.33, 
#                  callbacks = [es, mcp],
#                  verbose=1,
#                  )
# end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(test_csv)

# print("걸린시간: ", round(end_time - start_time, 2), "초")

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))  

# rmse = RMSE(y_test, y_predict)
# print("RMSE :", rmse)

submission['count'] = y_predict
# submission.to_csv(path + 'submit/submission_0911_MaxAbs.csv')

# #### plot 그리기 ####

# import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))

# plt.plot(hist.history['loss'][1:], c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'][1:], c = 'blue', label = 'val_loss')

# plt.title('kaggle bike loss')

# plt.xlabel('epoch')
# plt.ylabel('loss')

# plt.legend(loc = 'upper right')

# plt.grid()
# plt.show()


