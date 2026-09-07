import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


## california

from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

print(x_train.shape, y_train.shape)

# ## diabets

# from sklearn.datasets import load_diabetes

# datasets = load_diabetes()
# x = datasets.data
# y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)



# ## boston_housing

# from tensorflow.keras.datasets import boston_housing

# (x_train, x_test), (y_train, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)



model= Sequential()
model.add((Dense(5, input_dim=8)))
model.add(Dense(9))
model.add(Dense(13))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size = 4)

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
# results = model.predict(x_test)
# print("results: ", results)


from sklearn.metrics import r2_score, root_mean_squared_error

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
rmse = root_mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict)
