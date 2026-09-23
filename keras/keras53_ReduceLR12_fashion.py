from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

# 실습 맹그러봐요

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(np.unique(y_train, return_counts=True))



# one hot encoding
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델구성

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(10, (2,2), activation ='relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

model.add(Conv2D(12, (3,3), activation ='relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

model.add(Conv2D(24, (2,2), activation ='relu', padding = 'same'))
model.add(Conv2D(48, (2, 2), activation = 'relu', padding = 'same'))  
model.add(Conv2D(96, (2, 2), activation = 'relu', padding = 'same'))  
model.add(Conv2D(192, (2, 2), activation = 'relu', padding = 'same')) 


# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(10))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

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

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 10,
    restore_best_weights= True,
    verbose = 1
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 100000,
          batch_size = 128,
          verbose = 1,
          validation_split=0.2,
        #   metrics = ['acc'],
          callbacks = [es, rlr]
          )
end_time = time.time()




loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis =1)

acc_score = accuracy_score(y_pred, y_test) 

print("acc_score: ", acc_score)

print("걸린시간: ", round(end_time - start_time, 2), '초')




# maxpool 적용
# acc_score:  0.9034

## GAP 적용
# gap fasion mnist gpu에 
# acc_score:  0.9081
# 걸린시간:  797.26 초

# lr 0.01, patience 10
# acc_score:  0.8135
# 걸린시간:  69.96 초

# lr 0.01, patience 10, rlr
# acc_score:  0.8258
# 걸린시간:  94.03 초