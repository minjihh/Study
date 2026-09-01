import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # (10,)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1,],
             [9,8,7,6,5,4,3,2,1,0]]).transpose()   # (10,3) // x와 y의 데이터 개수가 동일하면 문제 없음
print(x.shape, y.shape) 


# [실습]
# 11, 0, -1 이 나오면 땡큐

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs=500, batch_size=2)


#4. 추론, 예측
loss = model.evaluate(x,y)
print("loss: ", loss)
results = model.predict(np.array([10]))
print("results: ", results)
