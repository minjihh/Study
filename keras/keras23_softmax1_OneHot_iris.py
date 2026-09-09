# 다중분류
# softmax, categorical cross entropy

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_iris()
print(datasets)
print(datasets.DESCR)  # 판다스의 describe 있음
print(datasets.feature_names) # 판다스의 ".columns" 똑같은 기능
# instance : 행
# 열, column, attribute, 속성

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)  # (150, 4) (150,)
print(y)
print(np.unique(y, return_counts=True)) # 이 코드를 쓰면 분류문제를 다룬다는 것 까지 예측할 수 있음

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=333,
    shuffle = True,   # shuffle을 false로 할 경우, 뒤쪽의 2는 훈련에 안들거아서 문제발생
    stratify = y,    # 

)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics = ['acc']) 
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=20,
    restore_best_weights=True,
)

start__time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
