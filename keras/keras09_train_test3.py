# 앞서했던 train/test 나누는 방식은 순차적으로 잘랐기 때문에 데이터의 편향을 일으킬 수 있음 (랜덤이 아님)
# scikit-learn을 활용해서 train/test 나눠보기

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
)

# x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=2024)

print("x_train: ", x_train)
print("x_test: ", x_test)
print("y_train: ", y_train)
print("y_test: ", y_test)

