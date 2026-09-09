from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48]))


y = pd.get_dummies(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(3, activation='softmax'))



#3. 컴파일, 훈련

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience =550,
    restore_best_weights=True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100000, batch_size=16, validation_split=0.2,  callbacks = [es]) 
end_time = time.time()

# 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss[0])
print("acc: ", round(loss[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score

accuracy_score = accuracy_score(y_test,y_pred)
print('acc_score: ', accuracy_score)

print("걸린시간: ", round(end_time - start_time,2), "초")

# acc = 0.95