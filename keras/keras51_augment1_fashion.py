# 50-2 카피
# randidx 생성해서 x_train의 랜덤한 인덱스값으로 랜덤한 train 데이터에 augment적용

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


########## 요기부터 증폿이닷 ##########

datagen = ImageDataGenerator(
    rescale = 1./255,  # 전처리
    horizontal_flip = True, # 수평 뒤집기
    # vertical_flip = True,  # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15, # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, # 좌표하나를 규정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode = 'nearest'  # 이미지를 수평이동 했을때 빈공간을 가까운 값으로 채우겠다.

)

augment_size = 40000

print(x_train.shape[0])
randidx = np.random.choice(x_train.shape[0], size = augment_size, replace = False)  # 6만개중에 4만개 랜덤뽑기 -> 숫자 60000 그대로 쓰는것보다 x_train.shape[0] 를 사용하면 편리
print(randidx)   # 벡터형태
print(randidx.shape)  # (40000,)   # randidx가 list형태 였다면 .shape로 확인 불가, list 의 경우 len으로 확인해야함
print(len(randidx))   #  40000

print(np.min(randidx), np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape)  # (40000, 28, 28) (40000,)
# x_train과 y_train 단순 복제상태

x_augmented = x_augmented.reshape(
    x_augmented.shape[0], 
    x_augmented.shape[1], 
    x_augmented.shape[2], 1)    # 4차원으로 바꾸기위해서 reshape

print(x_augmented.shape)  # (40000, 28, 28, 1)

xy_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size = augment_size,
                shuffle = False,
).next()[0]  # 신버전에서는 next(xy_augmented) 해줘야 실행됨
# 위 코드를 실행함으로서 x_augmented에도  dagaten이 적용됨어 아래서 xy_augmented[0] 말고 x_augmented로 사용해도 문제 없음



#### 변환 완료 ####
print(x_augmented.shape)  # (40000, 28, 28, 1)



datagen_origin = ImageDataGenerator(
    rescale = 1/255.
)

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

## 4차원으로

x_train = x_train.reshape(
    x_train.shape[0], 
    x_train.shape[1], 
    x_train.shape[2], 1)

x_test = x_test.reshape(
    x_test.shape[0], 
    x_test.shape[1], 
    x_test.shape[2], 1)


x_train = datagen_origin.flow(
                x_train, y_train,
                batch_size = 60000,
                shuffle = False,
).next()[0]   # next()[0] 으로 x값을 모두 가져온다. # [1]을하면 y값

# test = datagen_origin.flow(
#                 x_train, y_train,
#                 batch_size = 60000,
#                 shuffle = False,
# )   # AttributeError: 'NumpyArrayIterator' object has no attribute 'shape'
# next() 안쓰면 interator 상태

x_test = datagen_origin.flow(
                x_test, y_test,
                batch_size = 10000,
                shuffle = False,
).next()[0]


print(x_train.shape)    # (60000, 28, 28, 1)
print(y_train.shape)    # (60000,)

# exit()

print(x_test.shape)    # (10000, 28, 28, 1)


# exit()

print(x_train.shape)   # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)



# exit()

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)

print(np.unique(y_train, return_counts=True))



# one hot encoding
y_train = pd.get_dummies(y_train, dtype = int)
y_test = pd.get_dummies(y_test, dtype = int)


# [실습] 맹그러봐!!!

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(10, (2,2), activation ='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(12, (3,3), activation ='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation ='relu'))
model.add(Conv2D(16, (2, 2), activation = 'relu'))  
model.add(Conv2D(16, (2, 2), activation = 'relu'))  
model.add(Conv2D(16, (2, 2), activation = 'relu')) 


model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(10))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'] )

import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

pt_path = 'c:/Users/Admin/Desktop/Study_data/_save/keras51/'
filename = '{epoch:04d}-{val_loss:.4f}-fashion.keras'
filepath = "".join([pt_path, "k51_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath
)


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 100,
    restore_best_weights= True,
    verbose = 1,
   
)

start_time = time.time()
model.fit(x_train, y_train,
        epochs = 10000,
        batch_size = 64,
        validation_split = 0.2,
        callbacks = [es, mcp],
        verbose = 1
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




# acc_score:  0.8879
# 걸린시간:  1759.63 초

