import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

y = np.tanh(x) # -1 ~ 1

plt.plot(x, y)
plt.show()