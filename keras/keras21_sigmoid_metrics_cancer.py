# load_breast_cancer 데이터 불러오기
# type()를 통해 데이터 형태 확인
# numpy와 pandas 각각을 활용해서 레이블당 데이터 개수 확인

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer  # 유방암관련 데이터셋 불러오기

#1. 데이터 

datasets = load_breast_cancer()

### 보통은 데이터 판다스로 많이 불러오기 때문에 아래 2개랑 비슷한 역할하는 pandas 문법을 주로 사용 ###
print(datasets.DESCR)  # Number of Attributes: 30 numeric 
print(datasets.feature_names)
######################

# x = datasets.data   # datasets.data == datasets['data']
x = datasets['data']  # datasets은 딕셔너리형태의 데이터였다.
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))   # <class 'numpy.ndarray'>  type: data 형태 보여줌
# -> pandas 데이터도 numpy로 만들어져 있다.


# 분류형 모델 꼭 y를 확인해야함 -> label 종류 개수 확인필요
print(y)
# 0과 1의 개수가 몇개인지 찾아보기 -> numpy
print(np.unique(y))  # [0 1]
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357]))
# 분류/범주형 데이터 받으면 꼭 확인
# 범주형 데이터에서는 데이터 개수보다는 데이터 balance 좋은것이 좋다. 

# 0과 1의 개수가 몇개인지 찾아보기 -> pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size =0.5, random_state =1)

print(np.unique(y_train, return_counts=True))  
# (array([0, 1]), array([159, 239]))
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([53, 118]))



model = Sequential()
model.add((Dense(5, input_dim = 30)))
model.add((Dense(1)))

model.compile(loss='BCE', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es  = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True
)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100000, batch_size = 16, validation_data = (x_val, y_val))
end_time = time.time()

print("걸린시간: ", end_time - start_time)

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
results = model.predict(x_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))

plt.rcParams['font.family'] = 'Malgun Gothic'

plt.plot(hist['loss'], c = 'red', label = 'loss')
plt.plot(hist['val_loss'], c = 'blue', label = 'val_loss')

plt.title('breast_cancer_loss')

plt.legend(loc = 'upper right')

plt.xlabel('epoch')
plt.ylabel('loss')

plt.grid()
plt.show()
