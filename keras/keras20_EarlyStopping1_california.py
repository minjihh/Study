# 19-1 카피
# EarlyStopping 적용하기

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
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping  # EarlyStopping 카멜케이스

es = EarlyStopping(
    monitor='val_loss',
    mode = ' min',    # 사용하는 metric따라 다르게. accuracy라면 max가 좋음
    patience = 10,
    restore_best_weights=True,   # restore_best_weights 의 default 값은 False, 원칙은 True나 실제로 돌려보면 False일때 잘나오는 경우도 있음
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=500000, 
                 batch_size=4, 
                 verbose=1, 
                 validation_data = (x_val, y_val),
                 callbacks = [es], # early stopping 적용하기
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

# plt.rc('font', family='Malgun Gothic')   # 한글 폰트 깨질 때 
plt.rcParams['font.family']='Malgun Gothic'   # 한글 폰트 깨질 때, 바로 위 코드와 동일한 기능 

plt.figure(figsize=(9,6))    # 그래프 그릴판 사이즈

plt.plot(hist.history['loss'][1:], c='red', label='loss')    # y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][1:], c='blue', label='val_loss')  # epoch때 loss가 너무 크면 val_loss가 그래프상에서 0에 가깝게 나타나므로 1에폭을 스킵하고 그려서 볼 수 있도록 [1:] 설정

plt.legend(loc = 'upper right')    # 우측 상단에 라벨표시

plt.title('캘리포니아 Loss')
 
plt.xlabel('epoch')
plt.ylabel('loss')

plt.grid()    # 격자표시 추가
plt.show()







