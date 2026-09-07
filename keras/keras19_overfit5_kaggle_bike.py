import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
 
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.7, random_state=1)
print(x_train.shape, y_train.shape)  # (7620, 8) (7620,)


#2. 모델 구성

model = Sequential()
model.add(Dense(3, input_dim= 8))
model.add(Dense(1))


#3. 컴파일 훈련
### mse  ###
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs = 10, batch_size=4, validation_split=0.33)


#4. 평가 예측

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)




#### plot 그리기 ####

import matplotlib.pyplot as plt

plt.rcParams['font_family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')

plt.title('kaggle bike loss')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend(loc = 'upper right')

plt.grid()
plt.show()


#### R2 그리기 ###

import matplotlib.pyplot as plt

plt.rcParams['font_family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')

plt.title('kaggle bike R2_loss')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend(loc = 'upper right')

plt.grid()
plt.show()