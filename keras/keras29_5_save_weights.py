# 29-3 카피

from tensorflow.keras.models import Sequential, load_model
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()  # 절댓값이 가장 큰 값으로 모든 값을 나눈다.
scaler = RobustScaler()  # Standard scaling에 비해 분포의 범위가 넓어짐 -> 이상치의 영향을 덜받음, 이상치에 강함
# 실제 RobustScaler 사용할때는 이상치 처리를 먼저해준 다음에 scaling하는 방식으로 적용


# scaler.fit(x_train)    # sklearn에서 fit은 보통 실행하라는 의미, x_train scaling 비율에 맞게 ready
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)  # 위 두줄과 동일
x_test = scaler.transform(x_test)


print(x)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(1))

# model.summary()

path = './_save/keras29/'
# model.save(path + 'keras29_1_save_model.keras')   # 최신버전은 keras확장자. 옛날 모델은 h5 사용하기도 함
model.save_weights(path + 'keras29_5_save1.weights.h5')   


# model = load_model(path + 'keras29_1_save_model.keras')  # 여기서 저장하면 가중치 초기값이 저장됨

model.summary()

# exit()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    restore_best_weights= True
)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100000, batch_size=4, verbose=1, validation_data = (x_val, y_val),
        #   valdiatoin_split=0.5,
        callbacks = [es]
          )
end_time = time.time()

# model.save(path + 'keras29_3_save_model.keras')
model.save_weights(path + 'keras29_5_save2.weights.h5')  # weight만 저장할때는 확장자 weights.h5

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print("걸린시간: ", round(end_time - start_time, 2), "초")



def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# x 전체가지고 스케일링: 0.7308131612410732
# x_train만 가지고 MinMax 스케일링: 0.7333248910704415
# x_train만 가지고 Standard 스케일링: 0.7250561055506582
# x_train만 가지고 MaxAbs 스케일링: 0.7714653475245304
# x_train만 가지고 Robust 스케일링: 


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







