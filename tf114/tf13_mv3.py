import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error # = MSE

tf.compat.v1.set_random_seed(337)

x_data = [[1,2,3],
          [1,2,3]]
y_data = [[1],
          [2],
          [3]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')


#2. MODEL
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

# x.shape = (5, 3)
# y.shape = (5, 1)
# hy = x * w + b
#    = (5, 3) * w + b = (5, 1)
# (5, 3) * (?, ?) = (5, 1) 답 : (3, 1)

loss = tf.reduce_mean(tf.square(hypothesis)) # mse

#3. COMPILE, Fit
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, hy_val = sess.run([loss, w],
                              feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(epochs, 'loss:', cost_val)

y_predict = x * hy_val

print(sess.run([hypothesis, w, b], feed_dict={x:x_data}))
# print('mse: ', mean_squared_error(y_predict, y))
print('r2: ', r2_score(y_predict, y))
sess.close()
