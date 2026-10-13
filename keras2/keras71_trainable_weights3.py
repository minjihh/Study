# 71-2 카피

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

# 모델의 초기 가중치 고정
random.seed(333)
np. random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,])
y = np.array([1,])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))


model.summary()

######################## 동결 #######################
# model.trainable = False   # 동결 
model.trainable = True
######################## 동결 #######################
# model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, batch_size = 1, epochs = 100, verbose = 1)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

print("==========================================")
print(model.weights)
print("==========================================")


