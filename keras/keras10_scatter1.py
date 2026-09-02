# 09 test3 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

# [검색] train과 test를 섞어서 7:3 나눈다.
# 힌트 : scikitlearn


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,     # train 사이즈만 설정해도 test는 자동으로 나머지로 설정됨 // train_size=0.7 == test_size=0.3
    shuffle=True,       # shuffle은 설정하지 않아도 default=True   
    # train_size + test_size가 1보다 적어도 ok but 데이터를 쓰지 않으므로 손해
    # test_size default는 0.25이고 이때 train은 test의 나머지가 default로 설정 
    random_state = 333,
)

# x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=2024)

print("x_train: ", x_train)
print("x_test: ", x_test)
print("y_train: ", y_train)
print("y_test: ", y_test)

print(x_train.shape)


#2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

print("=============================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
results = model.predict(x_test)
print("results: ", results)


# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)   # 모든 데이터 점찍기

