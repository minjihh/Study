# 이진분류 데이터 다중분류 방식으로 풀어보기
# 이진분류로 했을때랑 비교해보기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.columns)


x = train_csv.drop(['target'], axis = 1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)
print(np.unique(y, return_counts=True))  # (array([0, 1]), array([179902,  20098]))

y = pd.get_dummies(y)


x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                  train_size=0.8,
                                                  random_state=1,
                                                  stratify=y)

model = Sequential()
model.add(Dense(5, input_dim=200, activation='relu' ))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


es = EarlyStopping(
    monitor='val_loss',
    mode ='auto',
    patience =100,
    restore_best_weights= True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=128, callbacks = [es])
end_time = time.time()

loss = model.evaluate(x_val, y_val)
print("loss: ", loss[0])
print("acc: ", loss[1])

print("걸린시간: ", round((end_time-start_time),2), "초")

y_pred = model.predict(test_csv)
y_pred = np.argmax(y_pred)


submission_csv['target'] = y_pred
submission_csv.to_cvs(path + "submission_0910_1032.csv")






