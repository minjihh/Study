# load_breast_cancer 데이터 불러오기
# type()를 통해 데이터 형태 확인
# numpy와 pandas 각각을 활용해서 레이블당 데이터 개수 확인
# train_test_split에서 stratify 설정 (분류모델일 경우)
# 이진분류 모델 마지막 layer에서 activaiton sigmoid 사용
# 모든 이진분류에서 loss BCE 사용, activation은 sigmoid
# compile에서 loss BCE로 변경, metrics=['acc'] 추가하여 훈련 로그에 출력되도록 
# accuracy로 훈련은 불가능

# sigmoid 함수 = 1 / (1 + e^(-z)) -> z가 무한대로 커질때: = e^(-z) 0으로 수렴 -> sigmoid: 0.9999...
# sigmoid 통과하고 나면 0과 1사이의 값이 나옴
# tensorflow에서는 sigmoid 적용후 반올림하도록 되어있음 -> 모델에서 sigmoid 통과후에는 0 또는 1의 값이 나옴 -> 이진분류


"""
            회귀        분류(이진)

Activation  linear     sigmoid
Loss         mse         BCE
round처리
(predict값)   X            O

* logistic regression은 회귀

"""


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state=1,
                                                    stratify=y,) 
# stratify 사용하면 원래 데이터 비율 유지
# stratify=y -> y의 클래스 비율을 유지하면서 분할
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size =0.5, random_state =1)

print(np.unique(y_train, return_counts=True))  
# (array([0, 1]), array([159, 239]))
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([53, 118]))

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

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
# model.add(Dense(5, input_dim = 30, activation= 'relu'))

# model.add(Dense(40, activation= 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation= 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation= 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation= 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(1, activation= 'sigmoid'))  # 통과후 0과 1사이의 값으로 나옴

# 마지막 layer의 activation은 sigmoid
# activation 명시하지 않을경우 default = linear

# 함수형

input1 = Input(shape=(30,))
dense1 = Dense(5)(input1)
dense2 = Dense(40)(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(40)(drop1)
drop2 = Dropout(0.5)(dense3)
dense4 = Dense(40)(drop2)
drop3 = Dropout(0.5)(dense4)
dense5 = Dense(40)(drop3)
drop4 = Dropout(0.5)(dense5)
output1 = Dense(1)(drop4)

model = Model(inputs = input1, outputs =output1)
model.summary()


#3. 컴파일 훈련
import datetime

model.compile(loss='binary_crossentropy', optimizer='adam',   # 이진분류 loss는 무조건 BCE
              # metrics = ['accuracy'],   # compile에 accuracy 추가함으로서 훈련로그에 accuracy가 같이 찍혀나오게 됨
              metrics = ['acc'],  # accuracy는 train, val data이용해서 각각 측정
              ) 
# 훈련시 출력되는 loss [BCE, Accuracy] 


es  = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,
    verbose=1
)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = './_save/keras34/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([pt_path, "k34_dropout_cancer", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100000, batch_size = 32, 
                 verbose=1,
                #  validation_data = (x_val, y_val), 
                 callbacks = [es, mcp], validation_split=0.3)
end_time = time.time()

print("걸린시간: ", round(end_time - start_time, 2))

loss = model.evaluate(x_test, y_test)


print("===============================")
# print("loss: ", loss)  # [loss, acc]
print("loss: ", loss[0]) 
print('acc:', round(loss[1], 4))  
print("===============================")


y_pred = model.predict(x_test)
# print(y_pred[:10])   # model 통과한후에는 값이 0과 1사이의 값으로 나옴, 결과값은 따로 반올림 필요
y_pred = np.round(y_pred) # y_pred 값 반올림 이용해서 0 또는 1 값으로 변환
print(y_pred[:10])


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", acc_score)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_pred)
print("RMSE :", rmse)
# x 전체가지고 스케일링: RMSE : 0.26490647239986226
# x_train만 가지고 MinMax 스케일링: RMSE : 0.25362863769573435
# x_train만 가지고 Standard스케일링: 0.22941573472520196
# x_train만 가지고 MaxAbs 스케일링: 0.21629522898011255
# x_train만 가지고 Robust 스케일링: 


# acc_score:  0.9590643274853801
# RMSE : 0.20232566030934981

# dropout 적용후
# acc_score:  0.9590643274853801
# RMSE : 0.20232566030934981

# 함수형
# acc_score:  0.005847953216374269
# RMSE : 10.85039147580504

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))

# plt.rcParams['font.family'] = 'Malgun Gothic'

# plt.plot(hist.history['loss'], c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')

# plt.title('breast_cancer_loss')

# plt.legend(loc = 'upper right')

# plt.xlabel('epoch')
# plt.ylabel('loss')

# plt.grid()
# # plt.show()
