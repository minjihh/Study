# 함수형 모델 (Model) 더 복잡한 모델을 만들때 유연하게 사용할 수 있음 (앙상블)

from tensorflow.keras.models import Sequential, Model    # Model: 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input

#2-1. 순차적 모델
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

###########################################################
#2-2. 함수형 모델
input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, name='ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
output1 = Dense(1)(drop2)

model2 = Model(inputs = input1, outputs = output1)

model2.summary()  # 함수형 모델의 경우 input_layer도 보여줌


