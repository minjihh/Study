# 코드실행으로 데이터 다운로드가 안되어 CIFAR10을 수동으로 파일을 받아서 옮김
# 'C:\Users\Admin\.keras\datasets'  경로에 파일 붙여넣기
# y데이터가 매트릭스 형태라서 pd.get_dummies() 적용하기 위해서 행렬 형태로 reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
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

### pd.get_dummies()  ###
y_train = y_train.reshape(-1,)  
y_train = pd.get_dummies(y_train, dtype=int)
print(y_train.shape)


y_test = y_test.reshape(-1,)
y_test = pd.get_dummies(y_test, dtype=int)
print(y_test.shape)

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
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same'))

model.add(Dense(40, activation ='relu'))
model.add(Dense(30, activation ='relu'))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation ='softmax'))

# from tensorflow.keras.layers import MaxPooling2D

# model.add(Conv2D(10, (2,2), input_shape = (10, 10, 1),
#                  strides=1,
#                  padding = 'same'))

# model.add(MaxPooling2D())
# model.add(Conv2D(filters = 9, kernel_size = (3,3),
#                  strides=1,
#                  padding = 'valid'))


#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics = ['acc'])


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 100000,
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

y_test = np.argmax(y_test, axis = 1)
y_pred = np.argmax(y_pred, axis = 1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score,2))

print("걸린시간: ", round(end_time - start_time, 2), '초')

# maxpool 적용
# acc_score:  0.76

## cpu에서 GAP 적용 -> 개선됨
# acc_score:  0.79
# 걸린시간:  3981.44 초

# lr 0.01, patience 10
# acc_score:  0.1
# 걸린시간:  62.76 초