import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

plt.plot(x, y)
plt.show()

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True)
plt.show()