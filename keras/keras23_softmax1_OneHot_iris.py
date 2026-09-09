# 다중분류
# softmax, categorical cross entropy
# tensorflow의 to_categorical사용해서 One Hot Encoding으로 데이터 변형
# 원핫인코딩 문제점: 메모리 많이 차지함, 0이 너무 많아짐, 성능저하 / 속도저하
# 이진분류 loss: BCE, 다중분류 loss: categorical crossentropy 다른건 없음

"""
                회귀     분류(이진)      분류(다중)
 One-Hot          X        X             O  

Activation
(마지막layer)    linear   sigmoid        softmax   

마지막layer
 node 개수        N         1            class개수
 
   loss          MSE       BCE       categorical_crossentropy
 
 predict       문제없음  np.round()     np.argmax()

"""

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

"""
One HotEncoding
벡터형태 y를 행렬 형태로 바꿈
[0, 0, 1, 1, 2]  # (5,)
->
[[1,0,0],
[1,0,0],
[0,1,0],
[0,1,0],
[0,0,1]]   # (5,3)

softmax사용해서 [1,1,0]과 같은 데이터 나오지 않도록 처리

"""
####################### 원핫1. to_categorial #######################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)   # (150, 3)

####################### 원핫2. padas #######################

y = pd.get_dummies(y)

####################### 원핫3. sklearn #######################
from sklearn.preprocessing import OneHotEncoder

y = OneHotEncoder(sparse=False).fit(y)


# 통상적으로 원핫인코딩 후에 trian_test_split 적용. train_test_split 먼저하면 원핫인코딩을 두번 해야함.
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=333,
    shuffle = True,   # shuffle을 false로 할 경우, 뒤쪽의 2는 훈련에 안들거아서 문제발생
    stratify = y,    # 

)

print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)  # (120, 3) (30, 3)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10))  # 중간에 activatoin 빼도 문제 없음
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))  # softmax -> 모든 값 더했을때 1넘지 않도록 함

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics = ['acc']) 
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()



#4. 평가, 예측
result = model.evaluate(x_test, y_test)  # softmax 통과까지 했지만 가장 높은값을 1로 바꾸는 작업은 아직 진행되지 않음
# print('loss: ', loss)   # loss가 2개 나옴, metrics =['acc']를 추가했기 때문에
print('loss: ', result[0])
print('acc: ', round(result[1], 2))

y_predict = model.predict(x_test)

### softmax 통과한 값 class label 값으로 바꿔주기 ###
import tensorflow as tf
# y_predict = tf.argmax(y_predict,1)
y_predict = np.argmax(y_predict, axis = 1) # [0 2 0 2 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
print(y_predict)

# y_test = tf.argmax(y_test,1)
y_test = np.argmax(y_test, axis = 1)       # [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
print(y_test)
###################################################


accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score: ', accuracy_score)
print('걸린시간: ', round(end_time - start_time, 2), '초')
# acc_score:  0.9666666666666667
# 걸린시간:  6.72 초



