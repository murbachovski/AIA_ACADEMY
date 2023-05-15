import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

tf.compat.v1.set_random_seed(1253)

# 1. DATA
x = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y = [[0], [0], [0], [1], [1], [1]]

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

logits = tf.compat.v1.matmul(xp, w) + b
hypothesis = tf.sigmoid(logits)

# 2-1. COMPILE
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yp, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss)

# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 2001
for step in range(epochs):
    cost_val = sess.run([loss, train], feed_dict={xp: x, yp: y})
    if step % 20 == 0:
        print(step, 'loss: ', np.round(cost_val[0], 3))

# 4. PREDICT
y_pred = sess.run(tf.round(hypothesis), feed_dict={xp: x})

accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
