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

#2. 모델 구성
model =Sequential()
model.add((Dense(3, input_dim=13)))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data= (x_val, y_val))
# model.fit(x_train, y_train, epochs=10, batch_size=4, validation_split= 0.33)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)


#### plot 그리기 ####

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')

plt.title('boston loss')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.legend(loc = 'upper right')

plt.grid()
plt.show()
