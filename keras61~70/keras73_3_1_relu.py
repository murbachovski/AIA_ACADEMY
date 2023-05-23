import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0, x)
# 위와 같다.
# relu = lambda x : np.maximum(0, x)

# x = np.arange(-5, 5, 0.1)
# y = relu(x)


# plt.plot(x, y)
# plt.show()


# 3_2, 3_3, 3_4...
# elu, selu, reaky_relu

import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def selu(x, alpha=1.6733, scale=1.0507):
    return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y_relu = relu(x)
y_elu = elu(x)
y_selu = selu(x)
y_leaky_relu = leaky_relu(x)
plt.plot(x, y_relu, label='RELU')
plt.plot(x, y_elu, label='ELU')
plt.plot(x, y_selu, label='SELU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def pelu(x, alpha=1.0, shared_parameter=True):
    if shared_parameter:
        return np.where(x >= 0, x, alpha * (np.exp(x / alpha) - 1))
    else:
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y_pelu_shared = pelu(x, shared_parameter=True)
y_pelu_individual = pelu(x, shared_parameter=False)

plt.plot(x, y_pelu_shared, label='PELU (Shared Parameter)')
plt.plot(x, y_pelu_individual, label='PELU (Individual Parameter)')
plt.legend()
plt.show()