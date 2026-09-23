# 실습
# 여자 데이터 증폭해서 성능 올려봐!!
# np.where 이용해서 여자 데이터만 분류
# concat후에 indice 섞어서 shuffle

# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 이진분류를 softmax로, 목표 acc 1.0

#1. 데이터


data_path = 'c:/Users/Admin/Desktop/Study_data/_data/image/faces/'

datagen = ImageDataGenerator(
    rescale = 1/255.
)

# xy = datagen.flow_from_directory(data_path,
#                                  target_size = (150, 150),
#                                  batch_size = 10000,
#                                  class_mode = 'binary',
#                                  color_mode = 'rgb',
#                                  shuffle = True,
# )

# x = xy[0][0]
# y = xy[0][1]

# print(x.shape, y.shape)  # (100, 200, 200, 3) (100, 3)

# print(np.unique(y, return_counts = True))  # (array([0., 1.], dtype=float32), array([6473, 3527], dtype=int64))

np_path = 'c:/Users/Admin/Desktop/Study_data/_data/gender_npy/'


x = np.load(np_path + 'keras46_gender_x.npy')
y = np.load(np_path + 'keras46_gender_y.npy')
# npy파일 one hot encoding 된 상태


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    random_state = 1,
                                                    stratify = y
                                                    )


x_train_w = x_train[np.where(y_train > 0.0)]
y_train_w = y_train[np.where(y_train > 0.0)]


# 확인
print(np.unique(y_train_w, return_counts=True))  # 1

########## 요기부터 증폿이닷 ##########

datagen = ImageDataGenerator(
    # rescale = 1./255,  # 전처리
    horizontal_flip = True, # 수평 뒤집기
    # vertical_flip = True,  # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    # height_shift_range=0.1, 
    rotation_range=15, # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,
    # shear_range = 0.7, # 좌표하나를 규정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode = 'nearest'  # 이미지를 수평이동 했을때 빈공간을 가까운 값으로 채우겠다.

)

augment_size = 8000
randidx = np.random.randint(x_train_w.shape[0], size = augment_size,)  # 6만개중에 4만개 랜덤뽑기 -> 숫자 60000 그대로 쓰는것보다 x_train.shape[0] 를 사용하면 편리

x_train_w_augmented = x_train_w[randidx].copy()
y_train_w_augmented = y_train_w[randidx].copy()

print(x_train_w_augmented.shape, y_train_w_augmented.shape)  # (8000, 150, 150, 3) (8000,)


xy_w_augmented = datagen.flow(
                x_train_w_augmented, y_train_w_augmented,
                batch_size = augment_size,
                shuffle = False,
).next()[0]  


# concat후 셔플 필요
x_train = np.concatenate((x_train, x_train_w_augmented))
y_train = np.concatenate((y_train, y_train_w_augmented))
print(x_train.shape, y_train.shape)  # (15000, 150, 150, 3) (15000,)

# 0부터 데이터 전체 길이까지의 무작위 인덱스 순서 생성
indices = np.random.permutation(len(x_train))
x_train = x_train[indices]
y_train = y_train[indices]


# print(np.unique(y_train_w, return_counts=True))
# # (array([0., 1.], dtype=float32), array([ 4568, 10432], dtype=int64))

# w와 m 합치기


# np_path = 'c:/Users/Admin/Desktop/Study_data/_data/gender_npy/'
# np.save(np_path + 'keras51_gender_x.npy', arr=x)
# np.save(np_path + 'keras51_gender_y.npy', arr=y)

# x = np.load(np_path + 'keras46_rps_x.npy')
# y = np.load(np_path + 'keras46_rps_y.npy')


# print(x_train.shape, y_train.shape)  # (15000, 150, 150, 3) (15000,)
# print(np.unique(y, return_counts = True))
# exit()

np_path = 'c:/Users/Admin/Desktop/Study_data/_data/keras51_faces_npy/'

# np.save(np_path + 'keras51_faces_x_train.npy', arr=x_train)
# np.save(np_path + 'keras51_faces_y_train.npy', arr=y_train)

# np.save(np_path + 'keras51_faces_x_test.npy', arr=x_test)
# np.save(np_path + 'keras51_faces_y_test.npy', arr=y_test)

# x_train = np.load(np_path + 'keras51_faces_x_train.npy')
# y_train = np.load(np_path + 'keras51_faces_y_train.npy')
# x_test = np.load(np_path + 'keras51_faces_x_test.npy')
# y_test = np.load(np_path + 'keras51_faces_y_test.npy')


#2. 모델구성

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (150,150,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
 

model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

# model.add(Dropout(0.2))

# model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
# model.add(Dropout(0.2))



# model.add(Flatten())
model.add(GlobalAveragePooling2D())   
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()



#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001  # keras default
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = 'binary_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime

path_pt = 'c:/Users/Admin/Desktop/Study_data/_save/keras52/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}-gender.keras'

file_path = "".join([path_pt, date,filename])

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    verbose = 1,
    factor = 0.5,  # lr을 50%로 줄이겠다.
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = file_path,
    save_best_only=True,
    verbose =1

)
es = EarlyStopping(
    monitor ='val_loss',
    mode = 'auto',
    verbose = 1,
    patience=50,
    restore_best_weights = True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs = 10000,
          batch_size = 16,
          validation_split=0.2,
          verbose =1,
          callbacks = [es, mcp, rlr]
          )
end_time = time.time()




#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)


acc_score =accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score, 2))
print("걸린시간: ", round(end_time - start_time, 2))


# acc_score:  0.88
# 걸린시간:  547.12

# 여자 data 증폭, patience = 100
# acc_score:  0.89
# 걸린시간:  3800.18

# 여자 data 증폭, concat후 shuffle, patience = 50
# acc_score:  0.87
# 걸린시간:  1203.87

# lr 0.01
# acc_score:  0.35
# 걸린시간:  581.63

# lr 0.01, rlr
# acc_score:  0.35
# 걸린시간:  688.16