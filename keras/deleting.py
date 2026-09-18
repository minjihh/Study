from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score

#1. 데이터

path_train = './_data/image/cat_dot/training_set'
path_test = './_data/image/cat_dot/test_set'

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    wiedth_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range = 0.7,
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size = 4000,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True,
)


xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size = (200, 200),
    batch_size = 1000,
    color_mode = 'rgb',
    shuffle = False,
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (200,200,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64,(3,3), activation = 'relu', padding= 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation = 'sigmoid'))

#3. 
model.compile()
model.fit