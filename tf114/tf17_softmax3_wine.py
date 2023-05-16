import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_diabetes, load_wine
import pandas as pd
import warnings

tf.compat.v1.set_random_seed(337)

x, y = load_wine(return_X_y=True)
print(x.shape, y.shape)
# (178, 13) (178,)
y_onehot = pd.get_dummies(y)

# 2. MODEL
x_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
w = tf.compat.v1.Variable(tf.random_normal([13, 3]))
b = tf.compat.v1.Variable(tf.zeros([3]), name='bias')

logits = tf.compat.v1.matmul(x_placeholder, w) + b
hypothesis = tf.nn.softmax(logits)

# 2-1. COMPILE
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=logits))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={x_placeholder: x, y_placeholder: y_onehot})
    if step % 200 == 0:
        print(step, val_loss)

# 4. PREDICT
y_pred = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x_placeholder: x})

acc = accuracy_score(y, y_pred)
print("accuracy_score:", acc)
# accuracy_score: 0.6741573033707865