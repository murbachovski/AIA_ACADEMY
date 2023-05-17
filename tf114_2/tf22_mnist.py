from keras.datasets import mnist
import keras
import tensorflow as tf
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

# Preprocess the data
num_classes = 10
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 2. MODEL
x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28])
y_input = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([28, 28]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([28]), name='bias1')
layer1 = tf.compat.v1.matmul(x_input, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([28, 14]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([14]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([14, num_classes]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([num_classes]), name='bias3')
logits = tf.compat.v1.matmul(layer2, w3) + b3
hypothesis = tf.compat.v1.matmul(layer2, w3) + b3

# 3-1. COMPILE
cost = tf.reduce_mean(tf.square(hypothesis - y_input))  # Use the reshaped y_input

# Print shapes for debugging
print(hypothesis.shape)
print(y_input.shape)
print(x_train.shape)
print(y_train.shape)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 3-2. PREDICT
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x_input: x_train, y_input: y_train})

        if step % 200 == 0:
            print(step, cost_val)

    h, a = sess.run([hypothesis, cost], feed_dict={x_input: x_train, y_input: y_train})
    print("예측값:", h, end='\n')
    print("MSE: ", a)

