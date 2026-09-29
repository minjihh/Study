import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 전처리
path = './_data/kaggle_jena/'
# jena_climate_2009_2016.csv 파일을 불러옵니다. 첫 번째 열(Date Time)을 인덱스로 사용합니다.
data_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# 마지막 144개 행(하루치 데이터, 10분 단위 * 6 * 24 = 144)을 제거합니다. 
# 문제 조건에 따라 마지막 144개 행은 제외합니다.
data = data_csv.iloc[:-144] 

# x(독립변수)와 y(종속변수) 분리
# 목표값인 'wd (deg)'(풍향, 제일 끝에 위치한 컬럼)을 y로 설정하고, 나머지 13개 특성을 x로 사용합니다.
x = data.drop(['wd (deg)'], axis=1)
y = data['wd (deg)']

# 모델 학습을 위해 pandas DataFrame/Series를 numpy 배열(array)로 변환합니다.
x = x.values
y = y.values

# 시계열 데이터를 RNN에 넣기 위해 일정 크기(timestep)로 분할하는 함수
# x 데이터를 size만큼 묶어서 하나의 샘플로 만듭니다.
def split_x(dataset, size):
    arr = []
    for i in range(len(dataset) - size):
        subset = dataset[i : (i + size)]
        arr.append(subset)
    return np.array(arr)

# 144 timestep (최근 144개의 시간 스텝 데이터를 보고 그 다음 스텝의 y를 예측)
size = 144

# x 데이터를 144개씩 묶어서 3차원 형태(데이터 개수, timestep, 피처 수)로 분할합니다.
x_split = split_x(x, size)

# y 데이터는 x 데이터 144개 바로 다음의 1개 값을 예측해야 하므로, size 번째 인덱스부터 가져옵니다.
y_split = y[size:]

print("x_split.shape:", x_split.shape) # 예측 시 (N, 144, 13) 형태가 됩니다.
print("y_split.shape:", y_split.shape) # 예측 시 (N,) 형태가 됩니다.

# 훈련 데이터와 테스트 데이터를 분리합니다. 
# 시계열 데이터이므로 과거의 데이터로 미래를 예측하도록 shuffle=False를 설정해 순서를 유지합니다.
x_train, x_test, y_train, y_test = train_test_split(
    x_split, y_split, train_size=0.8, shuffle=False
)

# 데이터 스케일링(정규화)을 위해 3차원인 x_train을 2차원으로 잠시 변환합니다.
# 3차원 상태로는 StandardScaler를 적용할 수 없기 때문에 (샘플 수 * timestep, 특성 수) 형태로 변경합니다.
x_train_reshape = x_train.reshape(-1, x_train.shape[2])
x_test_reshape = x_test.reshape(-1, x_test.shape[2])

# StandardScaler를 사용하여 특성(feature)들의 스케일을 동일하게 맞춥니다.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_reshape) # 훈련 데이터로 기준을 맞추고 변환
x_test_scaled = scaler.transform(x_test_reshape)       # 평가 데이터는 훈련 데이터의 기준을 그대로 적용

# 스케일링된 데이터를 다시 RNN 모델에 넣기 위해 본래의 3차원 형태(샘플 수, timestep, 특성 수)로 되돌립니다.
x_train = x_train_scaled.reshape(x_train.shape)
x_test = x_test_scaled.reshape(x_test.shape)


# 2. 모델 구성
model = Sequential()
# SimpleRNN 레이어: 
# input_shape=(144, 13) -> 144개의 시간적 연속성(timestep)과 13개의 변수(feature)를 의미합니다.
model.add(SimpleRNN(64, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
# 연속형 숫자 예측(풍향 각도)을 위한 회귀 모델이므로 마지막 출력 레이어는 노드 1개를 가지고 선형(기본) 활성화 함수를 사용합니다.
model.add(Dense(1)) 

model.summary()


# 3. 컴파일 및 훈련
# 회귀 문제이므로 손실 함수(loss)로 평균 제곱 오차(mse)를 사용합니다.
model.compile(loss='mse', optimizer='adam')

# 모델 학습 진행 
# epochs: 전체 데이터를 몇 번 반복해서 학습할지
# batch_size: 한 번에 학습할 데이터 묶음의 크기 (메모리 문제와 속도 향상을 위해 적절히 설정)
# validation_split: 훈련 데이터 중 일부를 쪼개어 학습 중간에 모델의 성능을 검증하는 데 사용합니다.
model.fit(x_train, y_train, epochs=10, batch_size=1024, validation_split=0.2)


# 4. 평가 및 예측
# 테스트 데이터를 사용해 학습된 모델을 평가하여 오차값을 구합니다.
loss = model.evaluate(x_test, y_test, batch_size=1024)
print("Loss (MSE):", loss)

# 학습이 끝난 모델을 바탕으로 테스트 세트의 일부(5개)에 대해 예측을 수행합니다.
y_pred = model.predict(x_test[:5])
print("실제값 (wd deg):", y_test[:5])
print("예측값 (wd deg):", y_pred.flatten())
