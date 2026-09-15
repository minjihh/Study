# 36-2 카피

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

#1. 데이터
(x_train, y_train,), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(y_train), np.max(y_train)) # 0 9

#### 스케일링 1 MinMax
# mnist -> minmax scaling이 적절. maxabs도 동일
# x_train = x_train/255.   # .을 붙임으로서 float 형태로 출력하게 됨
# x_test = x_test/255.

# print(np.min(x_train), np.max(x_train)) # 0.0 1.0
# print(np.min(x_test), np.max(x_test)) # 0.0 1.0

#### 스케일링 2 MaxAbs
x_train = (x_train - 127.5)/127.5   
x_test = (x_test- 127.5)/127.5

print(np.min(x_train), np.max(x_train)) # -1.0 1.0
print(np.min(x_test), np.max(x_test)) # -1.0 1.0



