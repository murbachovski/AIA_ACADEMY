import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

tf.compat.v1.set_random_seed(123)

# 1. DATA
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [154., 185., 180., 196., 142.]

# [실습]
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# 2. model
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)
# Calculation of R2 score
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# y_pred = sess.run(hypothesis, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
# r2 = r2_score(y_data, y_pred)

epochs = 2001
for step in range(epochs):
    cost_val, hy_val = sess.run([loss, train],
                              feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if step % 20 == 0:
        print(epochs, 'loss:', cost_val)

sess.close()


