# 78-5 카피
# elu -> 음수부분을 얼마까지 하한값을 alpha로 가질지 조정 가능

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
alpha = 1

# def elu(x, alpha):
#     return (x > 0)*x + (x<0)*(alpha*(np.exp(x)-1))

# exit()

### 연산량 효율은 swishsmith 

elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))

y = elu(x, alpha)

plt.plot(x,y)
plt.grid()
plt.show()


# print(y)


