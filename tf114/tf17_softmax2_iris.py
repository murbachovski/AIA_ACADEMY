import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

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
          [1,0,0]]

# 2. MODEL
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.random_normal([4, 3]))
b = tf.compat.v1.Variable(tf.zeros([1, 3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

logits = tf.compat.v1.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

# 2-1. COMPILE
loss = tf.reduce_mean(tf.nn.log_softmax(logits=logits))
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

# 3. FIT
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
    if step % 200 == 0:
        print(step, val_loss)

# 4. PREDICT
y_pred = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x: x_data})

acc = accuracy_score(np.argmax(y_data, axis=1), y_pred)
print("accuracy_score:", acc)
# accuracy_score: 0.42857142857142855