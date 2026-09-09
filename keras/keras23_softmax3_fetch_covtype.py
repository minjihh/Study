from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd

datasets = fetch_covtype()
# print(fetch_covtype.info())

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

exit()

# one hotencoding
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1, stratify=1)

model=Sequential()
model.add(Dense(5, input_dim=54, activation='relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

es = EarlyStopping(
    monitor = 'val_loss',
    mode='auto',
    patience = 250,
    restore_best_weights=True
)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
hist = model.fit(x_train, y_train, epochs= 100000, batch_size = 128, validation_split=0.2, callbacks=[es])
end_time = time.time()

loss = model.evaluate(x_train, y_train)
print("loss: ", loss[0])
print("acc: ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
acc_score = accuracy_score(y_test, y_pred)

print("acc_score: ", round(acc_score,2))
print("걸린시간: ", round(end_time - start_time, 2), "초")


# 시간재기
# 배치 크게
# acc 0.93