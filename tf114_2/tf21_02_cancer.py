import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer
import warnings

tf.set_random_seed(337)

# 1. DATA
x, y = load_breast_cancer(return_X_y=True)

print(x.shape, y.shape)
# (569, 30) (569,)

# Convert labels to one-hot encoded format
y_one_hot = np.zeros((y.shape[0], 2))
y_one_hot[np.arange(y.shape[0]), y] = 1

# 2. MODEL
x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 256]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([256]), name='bias1')
layer1 = tf.compat.v1.matmul(x_input, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([256, 128]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]), name='bias2')
layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128, 2]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([2]), name='bias3')
logits = tf.compat.v1.matmul(layer2, w3) + b3
hypothesis = tf.nn.sigmoid(logits)

# 3-1. COMPILE
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=logits))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 3-2. PREDICT
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x_input: x, y_input: y_one_hot})

        if step % 200 == 0:
            print(step, cost_val)

    h, p, a = sess.run([hypothesis, correct_prediction, accuracy], feed_dict={x_input: x, y_input: y_one_hot})
    print("예측값:", h, end='\n')
    print("모델값:", p)
    print("ACC: ", a)
# ACC:  0.65905094