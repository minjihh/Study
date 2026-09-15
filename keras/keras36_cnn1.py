# Conv2D 이미지 자르기
# 1 epoch에 convolution 최소 2개정도는 들어감
# vgg16 -> convolution 16개 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D 

model = Sequential()
model.add(Conv2D(10, (3,3), input_shape = (10,10,1)))   # (3,3) 커널 사이즈  # 10 x 10 사이즈의 흑백이미지
model.add(Conv2D(5, (2,2)))

model.summary()

# output shape: (None, 4, 4, 10)  입력 이미지 개수를 모르므로 None
# 이미지 가로 or 세로 - kernel 가로 or 세로 + 1  = 커널 통과후 가로 or 세로 사이즈

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 10)          100       
                                                                 
#  conv2d_1 (Conv2D)           (None, 7, 7, 5)           205       
                                                                 
# =================================================================
# Total params: 305
# Trainable params: 305
# Non-trainable params: 0
# _________________________________________________________________


