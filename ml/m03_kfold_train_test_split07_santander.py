
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.columns)


x = train_csv.drop(['target'], axis = 1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)
print(np.unique(y, return_counts=True))  # (array([0, 1]), array([179902,  20098]))

# y = pd.get_dummies(y, dtype=int)
# print(y)
# print(type(y))  # <class 'pandas.DataFrame'>


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                  train_size=0.8,
                                                  random_state=1,
                                                  stratify=y)


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# scaler = scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 333)

#2. 모델
model = XGBClassifier()


#3,4 훈련(컴파일 없음), 평가
scores= cross_val_score(model, x_train, y_train, cv = kfold)  # cross_val_score가 model.fit까지 시킴

print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

print(y_predict)
print(y_test)

# 맹그러바  accuracy_score

accuracy = accuracy_score(y_test, y_predict)
print("accuracy: ", accuracy)

# XGBClassifier
# accuracy:  0.9082