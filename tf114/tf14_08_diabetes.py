import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data_list = [
    load_diabetes(return_X_y=True),
    fetch_california_housing(return_X_y=True)
]
for i in data_list:
#1. DATA
    x, y = i
    # x, y = load_diabetes(return_X_y=True)
    # print(x.shape, y.shape) # (442, 10) (442,)
    # print(y[:10]) [151.  75. 141. 206. 135.  97. 138.  63. 110. 310.]
    y = y.reshape(-1, 1) # (442, 1)
    # x: (442, 10) * w: (?, ?) + b: (?) = y: (442, 1)   답: (10, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        random_state=337,
        train_size=0.8,
        test_size=0.2
    )

    print(x_train.shape, y_train.shape) # (353, 10) (353, 1)
    print(x_test.shape, y_test.shape) # (89, 10) (89, 1)

    xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
    yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 1]), name='weight')
    b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

    #2. MODEL
    hypothesis = tf.compat.v1.matmul(xp, w) + b

    #2-1. COMPILE
    loss = tf.reduce_mean(tf.square(hypothesis - yp)) # MSE
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    train = optimizer.minimize(loss)

    #3. FIT
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 2001
    for step in range(epochs):
        cost_val, hy_val = sess.run([loss, train], feed_dict={xp: x_train, yp: y_train})
        if step % 20 == 0:
            print(step)

    #4. PREDICT
    y_pred = sess.run(hypothesis, feed_dict={xp: x_test})

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

mse = mean_squared_error(y_test, y_pred)
print("MSE Score:", mse)