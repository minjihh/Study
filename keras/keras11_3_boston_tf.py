# boston_housing 데이터는 train/test로 구별되어 있음. 데이터 불러오는 부분 참고

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
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
results = model.predict(x_test)
print("results: ", results)




