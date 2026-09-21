import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.kears.preprocessing.image import load_img
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score


np_path = './_data/custom_image_npy/'
img = np.load(np_path + 'keras48_face.npy')

img = img/255.

model_path = './_save/keras46/'
model = load_model(model_path + '0921_15440017-0.4623-gender.keras')

result = model.predict(img)
print