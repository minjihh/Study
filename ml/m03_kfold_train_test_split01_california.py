# 05, 08 맹그러바

# 맹그러봐!!!


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1.데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
(x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.7, random_state=10)
print(x_train.shape, y_train.shape)  # (14447,8) (14447,)


n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 333)

#2. 모델
model = XGBRegressor()


#3,4 훈련(컴파일 없음), 평가
scores= cross_val_score(model, x_train, y_train, cv = kfold)  # cross_val_score가 model.fit까지 시킴

print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

print(y_predict)
print(y_test)

# 맹그러바  accuracy_score

accuracy = accuracy_score(y_test, y_predict)
print("accuracy: ", accuracy)

# XGBRegressor
# accuracy:  0.9