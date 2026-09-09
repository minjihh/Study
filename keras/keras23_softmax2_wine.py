from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

print(np.unique(y, axis=1))


y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13, activatoin='relu'))
model.add(Dense(40, activatoin='relu'))
model.add(Dense(40, activatoin='relu'))
model.add(Dense(40, activatoin='relu'))
model.add(Dense(40, activatoin='relu'))


#3. 컴파일, 훈련



# acc = 0.95