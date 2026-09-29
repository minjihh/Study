# 54-2카피
# LSTM은 RNN의 연산량의 4배 -> 느리다.
# 장기기억력은 LSTM이 RNN보다 좋다. 
# RNN 3차원 인풋
# x값 모델에 들어가기전에 reshape 주의

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

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
# model.add(SimpleRNN(2, input_shape = (3, 1)))   # input_shape = (timesteps, input_dim=feature)
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
# 파라미터 개수: units*features + units*bias + units*units
# timestpe이 3 -> hidden state도 3까지 존재
# model.add(LSTM(10, input_shape=(3,1)))
model.add(GRU(10, input_shape=(2,1)))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(1))

model.summary()




"""
# LSTM model.summary() # 

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10)                480       
                                                                 
 dense (Dense)               (None, 7)                 77        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0
_________________________________________________________________
"""

"""
# GRU model.summary() # 
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 10)                390 # tensorflow 특정 버전은 bias가 2개로 잡혀있음. 원래는 360이 맞음      
                                                                 
 dense (Dense)               (None, 7)                 77        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 475
Trainable params: 475
Non-trainable params: 0
_________________________________________________________________
"""