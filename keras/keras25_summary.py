# 지금껏 통상 y=w*x+b라고 했지만 실제로는 행렬/텐서의 연산이기 때문에 y = x*w + b 가 정석적인 형태

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np



#2. 모델
model = Sequential()
model.add(Dense(3, input_dim= 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary() # 파라미터 개수 세기
# h1 = w1*x +b 에서 각 weight는 연결된 선의 개수만큼, b는 선과 연결된 다음노드의 개수만큼이라고 계산할 수 있다.

