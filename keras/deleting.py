from tensorflow.keras.layers import Reshape
#2. 모델구성

model = Sequential()
model.add(Dense(280, input_shape=(28,28)))  # ( N, 28, 28) -> (N, 28, 280)
model.add(Reshape(target_shape = (28, 28, 10)))  # reshape -> 순서, 값이 바뀌면 안됨

model.add(Conv2D(64, (3,3), input_shape=(28, 28, 10)))  # (26,26,64) # conv2D에 넣기위한 4차원 데이터로 mnist 데이터의 shape 변환필요 # 첫번재 층에서는 linear
model.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', padding = 'same')) # (24,24,32)

# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation = 'relu'))  # units
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape = (32,), activation = 'relu'))
model.add(Dense(10, activation='softmax'))  # (10,)

model.summary()