import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


img_path = 'c:/Users/Admin/Desktop/Study_data/image/faces/'

datagen = ImageDataGenerator(
    rescale = 1/255.
)

xy = datagen.flow_from_directory(img_path,
                                 target_size = (200, 200),
                                 batch_size = 100,
                                 class_mode = 'binary',
                                 color_mode = 'rgb',
                                 shuffle = True,
)


x = xy[0][0]
y = xy[0][1]


np_path = './_data/gender_npy/'
np.save(np_path + 'keras44_gender_x.npy', arr=x)
np.save(np_path + 'keras44_gender_y.npy', arr=y)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    random_state = 1,
                                                    )


print(x_train.shape, y_train.shape)  #  (70, 200, 200, 3) (70,)



#2. 모델구성

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = (200,200,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))


model.add(Dense(64, activation = 'relu'))
model.add(Flatten())
# model.add(MaxPooling2D())   #     ValueError: Shapes (None, 3) and (None, 100, 100, 3) are incompatible
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


#3. 컴파일, 학습

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

path_pt = './_save/keras44/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}-gender.keras'

file_path = "".join([path_pt, date,filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    filepath = file_path,
    save_best_only=True,
    verbose =1

)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 100,
    restore_best_weights= True,
    verbose =1
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 100000,
          batch_size = 16,
          callbacks = [es, mcp],
          verbose = 1,
          validation_split = 0.2)
end_time = time.time()


#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score: ", round(acc_score, 2))

print("걸린시간: ", round(end_time - start_time, 2) , '초')