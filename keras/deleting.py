from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_trani), (x_test, y_test) = fashion_mnist.load_data()



datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range = 0.1,
    height_shift_range=  0.1,
    rotation_range = 15,
    zoom_range = 1.1,
    shear_range = 0.7,
    fill_mode = 'nearest'
)

augment_size = 100

print(x_train.shape)
print(x_train[0].shape)

aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape)


xy_data = datagen.flow(
    np.tile(x_train[0], augment_size).reshape(-1,28,28,1),
    np.zeros(augment_size),
    batch_size = augment_size,
    shuffle=False,
).next()

print(xy_data)
print(len(xy_data)) 


print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100,) 

plt.figure(figsize = (7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(xy_data[0][1], cmap = 'gray')
plt.show()
