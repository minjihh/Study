import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score


from sklearn.datasets import load_iris

datasets = load_iris()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)

print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                   random_state=1,
                                                   )

# onehotencoding 적용
from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

# y = pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# y = ohe.fit_tranform(y)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10))  # 중간에 activatoin 빼도 문제 없음
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))  # softmax -> 모든 값 더했을때 1넘지 않도록 


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    restore_best_weights= True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs =10000,
                 batch_size = 16,
                 callbacks =[es]
                 )
end_time = time.time()

loss = model.evaluate(x_test, y_test)
print("loss: ", loss[0])
print("acc: ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)









