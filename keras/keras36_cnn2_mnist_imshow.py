import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train,), (x_test, y_test) = mnist.load_data()

# print(x_train)
# print(x_train[0])

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
# reshape 조건 : 1) 순서 바뀌지 않고, 2) 값도 바뀌면 안됨

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

print(pd.value_counts(y_test))

import matplotlib.pyplot as plt

plt.imshow(x_train[50],) # 'gray')  # 원래 grayscale 이미지에서 'gray' 설정 안했다고해서 color 이미지라고 할 순 없음
plt.show()
