import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6

# def f (x):
#     return x**2 -4*x + 6

gradient = lambda x : 2*x - 4

x = -10.0 # 초기값
epochs = 20
learning_rate = 0.25

x_values = []
y_values = []

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    y = f(x)
    x_values.append(x)
    y_values.append(y)
    print(i + 1, x, y)

# Plotting the function
x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label='f(x)')
plt.plot(x_values, y_values, 'ro-', label='Optimization Process')
plt.plot(2, f(2), 'sk', label='Minimum')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()