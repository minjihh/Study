# 앞서했던 train/test 나누는 방식은 순차적으로 잘랐기 때문에 데이터의 편향을 일으킬 수 있음 (랜덤이 아님)



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 


