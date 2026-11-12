# 2017년 구글에서 swish 발표
# Sigmoid Linear Unit 
# silu -> x에 sigmoid 곱한것

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def silu(x):
    return x * (1 / (1 + np.exp(-x)))   # 음수의 0 부분 0 처리하기 찜찜해서 제안됨, 


silu = lambda x: x* (1 / (1 + np.exp(-x)))

y = silu(x)

plt.plot(x,y)
plt.grid()
plt.show()


# print(y)


