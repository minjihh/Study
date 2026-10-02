# 36-6 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

# 실습 맹그러봐!
# 0.67

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(np.unique(y_train, return_counts=True))   
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))

# print(y_train)


#OHE

# ### pd.get_dummies()  ###
# y_train = y_train.reshape(-1,)
# y_train = pd.get_dummies(y_train, dtype=int)
# print(y_train.shape)


# y_test = y_test.reshape(-1,)
# y_test = pd.get_dummies(y_test, dtype=int)
# print(y_test.shape)

### sklearn ###
# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(-1,1)
# y_train = ohe.fit_transform(y_train)
# y_test = y_test.reshape(-1,1)
# y_test = ohe.fit_transform(y_test)



#2. 모델구성

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape = (32, 32, 3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))




model.add(Dense(40, activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation ='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation ='softmax'))


#3. 컴파일 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=500,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 1000,
          batch_size = 128,
          validation_split=0.2,
          verbose = 1,
          callbacks = [es]
          )
end_time = time.time()


#4. 평가 예측
print("=============== model.evaluate ===============")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_pred = model.predict(x_test)

# y_test = np.argmax(y_test, axis = 1)
y_pred = np.argmax(y_pred, axis = 1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score,2))

print("걸린시간: ", round(end_time - start_time, 2), '초')


# 0.4
# acc_score:  0.21
# acc_score:  0.17
# acc_score:  0.18
# acc_score:  0.19