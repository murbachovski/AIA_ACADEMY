import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

tf.compat.v1.set_random_seed(337)

x_data = [[1, 2, 3], [1, 2, 3]]
y_data = [[1], [2], [3]]
x_data = x_data + [[1, 2, 3]]
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

# 2. MODEL
hypothesis = tf.compat.v1.matmul(x, w) + b

# 3. COMPILE, Fit
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
sess.run(hypothesis, feed_dict={x: x_data})

# Get predicted values
y_pred = sess.run(hypothesis, feed_dict={x: x_data})

# Calculate R2 score
r2 = r2_score(y_data, y_pred)
print("R2 Score:", r2)
mse = mean_squared_error(y_data, y_pred)
print("MSE Score:", mse)
