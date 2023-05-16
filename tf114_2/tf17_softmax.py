import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

tf.set_random_seed(337)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

#2. MODEL
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]))
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name = 'bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis = tf.matmul(x, w) + b
logits = tf.compat.v1.matmul(x, w) + b

#2-1. COMPILE
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # MSE
loss = tf.reduce_mean(tf.nn.softmax(logits=logits))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
# train = optimizer.minimize(loss)
# 한 줄로 요약 가능합니다.
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 2001
for step in range(epochs):
    val_loss = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print(step, val_loss[0])

# 4. PREDICT
y_pred = sess.run(tf.round(hypothesis), feed_dict={x: x_data})

# mae = mean_squared_error(y_data, y_pred)
# print("mean_squared_error:", mae)