# 9-1 카피
# model.fit에서 verbose 파라미터 설정하는 법
# verbose를 설정하는 것 자체도 delay

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=2,
            verbose=4, 
          )  
# verbose 파라미터: 훈련되는 과정을 terminal에 나오지 않도록 설정
# verbose = 0: 침묵
# verbose = 1: 디폴트 (훈련과정 다 출력)
# verbose = 2: 프로그래스 바 생략하고 epoch, los 값만 나타냄
# verbose = 3: epoch만 나옴
# verbose = 나머지: epoch만 나옴
# verbose 설정 다르게 하는 이유: progress바, epoch등을 표시하기 위해서 원래 연산은 더 빨리 끝났지만 machine이 delay하고 있음
# verbose 설정으로 인해 모델 성능에 영향은 없음
# 1epoch에 batch 4로 한번 batch 3으로 한번 총 두번돌음

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
# results = model.predict(x_test, y_test)
# print("results: ", results)


