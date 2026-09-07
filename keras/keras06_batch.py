# 훈련횟수가 늘어날수록 모델의 성능이 좋아짐. (9000장 한번 훈련보다 3000장 세번 훈련이 더 좋다.)
# 배치를 자를때 랜덤하게 조합해서 과적합을 방지할 수 있음.
# 데이터개수가 너무 적을 때는 통배치로 학습하는게 나을수도 있다.
# 2개이상 list

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,4,3,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=2)    # 데이터 6개일때 batch_size 4로 설정하면 4개/2개 순서로 훈련됨, tensorflow default batch size: 32
 
#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss )
# result = model.predict(np.array([1,2,3,4,5]))
# print("7의 예측값 : ", result)