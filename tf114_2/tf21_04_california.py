import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import warnings

tf.set_random_seed(337)

# 1. DATA
x, y = fetch_california_housing(return_X_y=True)

print(x.shape, y.shape)
# (20640, 8) (20640,)

# Convert the shape of y to (20640, 1)
y = y.reshape(-1, 1)

# 2. MODEL
x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 256]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([256]), name='bias1')
layer1 = tf.compat.v1.matmul(x_input, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([256, 128]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128, 1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias3')
logits = tf.compat.v1.matmul(layer2, w3) + b3
hypothesis = tf.compat.v1.matmul(layer2, w3) + b3

# 3-1. COMPILE
cost = tf.reduce_mean(tf.square(hypothesis - y_input)) # MSE
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 3-2. PREDICT
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x_input: x, y_input: y})

        if step % 200 == 0:
            print(step, cost_val)

    h, a = sess.run([hypothesis, cost], feed_dict={x_input: x, y_input: y})
    print("예측값:", h, end='\n')
    print("MSE: ", a)
