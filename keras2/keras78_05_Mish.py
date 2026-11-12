# 78-4 카피

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def mish(x):
#     return x * np.tanh(np.log(1+ np.exp(x)))

# exit()



### 연산량 효율은 swishsmith
# 
#  

mish = lambda x: x * np.tanh(np.log(1+ np.exp(x)))

y = mish(x)

plt.plot(x,y)
plt.grid()
plt.show()


# print(y)


