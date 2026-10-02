# 36-3 copy
# 코드실행으로 데이터 다운로드가 안되어 CIFAR10을 수동으로 파일을 받아서 옮김
# 'C:\Users\Admin\.keras\datasets'  경로에 파일 붙여넣기
# y데이터가 매트릭스 형태라서 pd.get_dummies() 적용하기 위해서 행렬 형태로 reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

# 실습 맹그러봐!
# 0.67

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(np.unique(y_train, return_counts=True))   
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))


#OHE

# ### pd.get_dummies()  ###
# y_train = y_train.reshape(-1,)   # (50000, 1) 을 (50000,) 으로 reshape
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

### tensorflow Onehoeencoding ###
# from tensorflow.keras.utils import to_categorical

# y_train  = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = (32, 32, 3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(Dense(40, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Flatten())
model.add(Dense(10, activation ='softmax'))




#3. 컴파일 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=100,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 10,
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

