# padding 적용
# kernel이 convolution할때 중첩이 없는 세팅은 권하지 않음
# shape 유지하고 싶을때 -> 패딩
# stride 크게줘서 사이즈 확 줄일 수 있음

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape = (10, 10, 1 ),   # 10, 10, 10
                 strides = 1,  # stride '1' default
                 padding = 'same',  
                   ))
# 통상적으로 convolution 이후에 max pooling 사용
model.add(MaxPooling2D())  # max pool 써서 size줄일때와 같은 효과를 convolution으로 내기 위해서는 훨씬 더 많은 parameter가 필요

model.add(Conv2D(filters = 9, kernel_size = (3,3),  # 8, 8, 9
                 strides = 1,
                 padding = 'valid',  # default: 'valid'
                 ))

model.summary()


