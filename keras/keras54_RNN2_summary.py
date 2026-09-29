# 54-1카피
### input data ###
# DNN: 2차원데이터
# RNN: 3차원 데이터
# CNN: 4차원 데이터
# RNN -> 파라미터수가 input/output size에 영향을 받지 않음. 매 time step마다 동일한 param 사용
# h1값은 x1에서 h1나오기위해 통과한 layer, h2를 출력하는 layer, h3를 출력하는 layer 3번 통과
# h2값도 역시 h1, h2, h3 출력하는 layer를 통과한다. (h2가 출력되기전에 이전에 한번 통과했던 h1 출력 layer 다시한번 통과)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

# (10,) 벡터데이터를 (7,3) 매트릭스로 바꿈

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)  # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)  # (7,3,1) = (sample, timesteps, feature)
print(x.shape)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3, 1)))
model.add(SimpleRNN(10, input_shape = (3, 1)))   # input_shape = (timesteps, input_dim=feature)
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# 파라미터 개수: units*features + units*bias + units*units
# timestpe이 3 -> hidden state도 3까지 존재
# RNN은 activation tahh가 unit * unit 만큼 존재함 (hidden쪽)
model.add(Dense(7, activation = 'relu'))
model.add(Dense(1))

model.summary()




"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 5)                 35        
                                                                 
 dense (Dense)               (None, 7)                 42        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 85
Trainable params: 85
Non-trainable params: 0
_________________________________________________________________ 
"""

