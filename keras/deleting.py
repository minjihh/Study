# 50-2 카피

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()


datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    width_shift_range = 0.1,
    rotation_range=15,
    fill_mode = 'nearest'
)

augment_size = 40000

print(x_train.shape[0])
randidx = np.random.choice(x_train.shape[0], size = augment_size, replace= False)
print(randidx)
print(randidx.shape)
