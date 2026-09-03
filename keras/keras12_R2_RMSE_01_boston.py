# 11_3 카피
# RMSE: MSE에서 error값이 큰경우 조정하기 위해서 mse에 root값을 씌움
# R2 (R squared) = 1 - (MSE)/Var(y)d: loss로 판단이 안될경우 보조 지표로 활용, 우선적으로 확인할 것은 loss // 회귀모델에서 사용하며 1에 가까울수록 모델 성능이 좋음
# Tensorflow에 R2가 없음

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)


#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim =13))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # 훈련할 때마다 loss 계산 // loss 값 비교해서 역전파 -> weight update
model.fit(x_train, y_train, epochs=20, batch_size=4)

print("================================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics


# print("results: ", results)




