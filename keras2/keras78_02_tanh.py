import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)  # -1에서 1사이의 값을 가짐

plt.plot(x, y)
plt.grid()
plt.show()

