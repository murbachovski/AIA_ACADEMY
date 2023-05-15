import tensorflow as tf
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

tf.compat.v1.set_random_seed(1253)

#1. DATA
x = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y = [[0], [0], [0], [1], [1], [1]]

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

hypothesis = tf.compat.v1.matmul(xp, w) + b

#2-1. COMPILE
loss = tf.reduce_mean(tf.square(hypothesis - yp)) # MSE
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.09)

train = optimizer.minimize(loss)

#3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    sess.run([loss, train], feed_dict={xp: x, yp: y})
    if step % 20 == 0:
        print(step)

#4. PREDICT
y_pred = sess.run(hypothesis, feed_dict={xp: x})
print(type(y_pred))

r2 = r2_score(y, y_pred)
print("R2 Score:", r2)

mse = mean_squared_error(y, y_pred)
print("MSE Score:", mse)

# R2 Score: 0.8837209519489212
# MSE Score: 0.02906976201276971