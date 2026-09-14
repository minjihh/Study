# 30-1 카피
# drop out : 모델쪽에서 과적합 방지할 수 있는 방법
# 추론시에는 dropout 적용하면 안됨

# ##데이터셋
# 01 california 
# 02 diabetes
# 03 boston

# 04 dacon_ddarung
# 05 kaggle bike
# 06 cancer
# 07 snatander

# 08 wine
# 09 fetch_convtype 
# 10 digits

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


path = './_save/keras34/'


#1. 데이터

datasets = fetch_california_housing()   # 회귀 데이터
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
# 통상적으로 dropout 0.5이상을 잘 주지 않음
# model = Sequential()
# model.add(Dense(5, input_dim=8, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(40, activation = 'relu'))
# model.add(Dropout(0.3))

# model.add(Dense(40, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation = 'relu'))

# model.add(Dense(1))

# 함수형
input1 = Input(shape=(8,))
dense1 = Dense(5)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(40)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(40)(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(40)(drop3)
output1 = Dense(1)(dense4)

model = Model(inputs = input1, outputs = output1)
model.summary()

#3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model.compile(loss='mse', optimizer='adam')


# es와 mcp도 verbose옵션을 사용해서 es나 mcp가 걸리고 있는지 상태 확인가능
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights= True,
    verbose=1, 
)

mcp = ModelCheckpoint(
    monitor='val_loss',    # loss로 해도 상관없음
    mode = 'auto',
    save_best_only = True,
    filepath = path + 'keras34_california.keras',
    verbose = 1,
)



start_time = time.time()
hist = model.fit(x_train, y_train,
                epochs=1000, batch_size=32, verbose=1, validation_data = (x_val, y_val),
        #   valdiatoin_split=0.5,
                callbacks = [es, mcp]
          )
end_time = time.time()

#4. 평가, 예측
mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print("걸린시간: ", round(end_time - start_time, 2), "초")

print("mse: ", mse)

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# x 전체가지고 스케일링: 0.7308131612410732
# x_train만 가지고 MinMax 스케일링: 0.7333248910704415
# x_train만 가지고 Standard 스케일링: 0.7250561055506582
# x_train만 가지고 MaxAbs 스케일링: 0.7714653475245304
# x_train만 가지고 Robust 스케일링: 


# mse:  0.2908291220664978
# r2:  0.7782544566970491
# RMSE : 0.5392857528562534

## dropout 적용후 ##
# mse:  0.4394420385360718
# r2:  0.664943073293314
# RMSE : 0.6629042407562711

# 함수형
# mse:  1.2540855407714844
# r2:  0.043810157603821454
# RMSE : 1.1198595551093624


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







