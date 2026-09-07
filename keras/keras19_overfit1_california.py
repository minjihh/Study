# 17-1 카피
# hist.history를 통해 loss, val_loss이 리스트를 반환할 수 있다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)

print(x_train.shape, y_train.shape)   # (14447, 8) (14447,)

#2. 모델구성
model =Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=4, verbose=1, validation_data = (x_val, y_val)
        #   valdiatoin_split=0.5
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print("===================== history ========================")
print(hist)
print("===================== hist.history ========================")
print(hist.history)
print("========================= loss ============================")
print(hist.history['loss'])    # hist.history 딕셔너리에 있는 loss값만 빼자.
print("======================== val_loss ===========================")
print(hist.history['val_loss'])    # hist.history 딕셔너리에 있는 val_loss값만 빼자.
print("============================================================")



import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))    # 그래프 그릴판 사이즈

plt.plot(hist.history['loss'][3:], c='red', label='loss')    # y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][3:], c='blue', label='val_loss')

plt.legend(loc = 'upper right')    # 우측 상단에 라벨표시

plt.rc('font', family='Malgun Gothic') 

# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False
plt.title('캘리포니아 Loss')
 
plt.xlabel('epoch')
plt.ylabel('loss')

plt.grid()    # 격자표시 추가
plt.show()







