# 19-1 카피
# 스케일링 -> 모든 x값에 대해서 같은 비율로 처리하면 데이터 조작 문제 없음
# x의 값이 너무 클 경우, 역전파할때도 스케일링 하는게 성능향상에 도움될 수 있음
# 역전파시 가장 잘되는 데이터 -> 부동소수점 형태
# x값은 0~1사이 값으로 바꾸고 싶을때 -> (x-min)/(max-min) (=minmax scaler)
# 
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

"""
MinMaxScaler 

원값 - Min
-----------
Max - Min

"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)    # sklearn에서 fit은 보통 실행하라는 의미
x = scaler.transform(x)

print(x)
print(np.min(x), np.max(x))
# 0.0 1.0000000000000002


exit()




#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1, validation_data = (x_val, y_val)
        #   valdiatoin_split=0.5
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")

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

# plt.rc('font', family='Malgun Gothic')   # 한글 폰트 깨질 때 
plt.rcParams['font.family']='Malgun Gothic'   # 한글 폰트 깨질 때, 바로 위 코드와 동일한 기능 

plt.figure(figsize=(9,6))    # 그래프 그릴판 사이즈

plt.plot(hist.history['loss'][2:], c='red', label='loss')    # y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')

plt.legend(loc = 'upper right')    # 우측 상단에 라벨표시

plt.title('캘리포니아 Loss')
 
plt.xlabel('epoch')
plt.ylabel('loss')

plt.grid()    # 격자표시 추가
plt.show()







