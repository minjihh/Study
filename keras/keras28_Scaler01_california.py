# 27-1 카피
# 사실 27-1 코드는 틀린코드 
# train_test_split 후에 해줘야함
# 스케일링 -> 모든 x값에 대해서 같은 비율로 처리하면 데이터 조작 문제 없음
# x의 값이 너무 클 경우, 역전파할때도 스케일링 하는게 성능향상에 도움될 수 있음
# 역전파시 가장 잘되는 데이터 -> 부동소수점 형태
# x값을 0~1사이 값으로 바꾸고 싶을때 -> (x-min)/(max-min) (=minmax scaler)
# x_train만 가지고 scaling x_test는 나중에 추론시 x_train에 적용했던 스케일링 비율 그대로 x_test에 적용
# 이에따라 x_test는 1이 넘어가거나 0보다 작은값이 나올 수 있음 
# x_test까지 scaling 비율에 이용할 경우 과적합의 위험 있음

# scaling 하는 이유: 특정 피처가 큰값을 가지는 경우 과도하게 영향을 미치는 것 방지
# 컬럼별로 scaling 적용 -> 컬럼별로 동등하게 영향을 미치도록

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=1)
print(x_train.shape, y_train.shape)   # (14447, 8) (14447,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)    # sklearn에서 fit은 보통 실행하라는 의미, x_train scaling 비율에 맞게 ready
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# exit()


"""
MinMaxScaler 

원값 - Min
-----------
Max - Min

"""

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 100,
    restore_best_weights= True
)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100000, batch_size=4, verbose=1, validation_data = (x_val, y_val),
        #   valdiatoin_split=0.5,
        callbacks = [es]
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")



def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# x 전체가지고 스케일링: 0.7308131612410732
# x_train만 가지고 스케일링: 0.7333248910704415




# print("===================== history ========================")
# print(hist)
# print("===================== hist.history ========================")
# print(hist.history)
# print("========================= loss ============================")
# print(hist.history['loss'])    # hist.history 딕셔너리에 있는 loss값만 빼자.
# print("======================== val_loss ===========================")
# print(hist.history['val_loss'])    # hist.history 딕셔너리에 있는 val_loss값만 빼자.
# print("============================================================")



# import matplotlib.pyplot as plt

# # plt.rc('font', family='Malgun Gothic')   # 한글 폰트 깨질 때 
# plt.rcParams['font.family']='Malgun Gothic'   # 한글 폰트 깨질 때, 바로 위 코드와 동일한 기능 

# plt.figure(figsize=(9,6))    # 그래프 그릴판 사이즈

# plt.plot(hist.history['loss'][2:], c='red', label='loss')    # y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')

# plt.legend(loc = 'upper right')    # 우측 상단에 라벨표시

# plt.title('캘리포니아 Loss')
 
# plt.xlabel('epoch')
# plt.ylabel('loss')

# plt.grid()    # 격자표시 추가
# plt.show()







