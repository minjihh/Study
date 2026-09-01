import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense     # from 안에 뭐들었는지 궁금하면 ctrl + space
import numpy as np


#1. 데이터

# x가 (2,5), y는 (5,)라서 아래 데이터로는  학습 불가. x(5,2), y(5,)이거나 x(2,5), y(2,) 이어야 학습 가능
# x = np.array([[1,2,3,4,5],
#               [6,7,8,9,10]])
# y = np.array([1,2,3,4,5])


# 위에 데이터를 올바르게 수정한 아래 데이터. // 열=column=피처
x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]])
y = np.array([1,2,3,4,5])


print(x.shape)  # (5,2)
print(y.shape)  # (5,)