import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # [0 1 2 3 4 5 6 7 8 9]
print(x)

x = np.array(range(1,10)) # [1 2 3 4 5 6 7 8 9]
print(x)

x = np.array(range(1,11)) # [ 1  2  3  4  5  6  7  8  9 10]
print(x)

x = np.array([range(10), range(21, 31,), range(201, 211)]).T # 각 range로 생성되는 갯수가 같으므로 문제 없는 데이터. 3x10 행렬 형태 // x=x.T와 동일
print(x)
print(x.shape)  # (3,10)  -> (10,3)

y = np.array(range(1,11))
print(y.shape)

#2. 모델구성
# [실습]
# [10, 31, 211]

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x,y,epochs=500, batch_size=2)

#. 평가, 예측
loss = model.evaluate(x,y)
print("loss: ", loss)
results = model.predict(np.array([[10, 31, 211]]))
print("results: ", results)