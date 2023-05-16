import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

tf.compat.v1.set_random_seed(1253)

# 1. DATA
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1253)

y_train = np.reshape(y_train, (-1, 1))

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, x.shape[1]])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x.shape[1], 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

logits = tf.compat.v1.matmul(xp, w) + b
hypothesis = tf.compat.v1.sigmoid(logits)

# 2-1. COMPILE
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yp, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 2001
for step in range(epochs):
    val_loss = sess.run([loss, train], feed_dict={xp: x_train, yp: y_train})
    if step % 20 == 0:
        print(step, val_loss[0])

# 4. PREDICT
y_pred = sess.run(tf.round(hypothesis), feed_dict={xp: x_test})

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
