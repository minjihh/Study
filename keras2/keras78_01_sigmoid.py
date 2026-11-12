# weight 초기값은 통상 0과 1사이의 값, bias 초기값은 0
# 마지막 층의 activation은 relu 사용하지 않음

import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
    # return 1 / (1 + np.exp(-x))

sigmoid = lambda x: 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))  # 100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()




