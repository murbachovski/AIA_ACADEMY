import tensorflow as tf
tf.compat.v1.set_random_seed(337)

x_data = [[1,2,3],
          [1,2,3]]
y_data = [[1],
          [2],
          [3]]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([]), name='bias')


#2. MODEL
hypothesis = x * w + b

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print(sess.run([hypothesis, w, b], feed_dict={x:x_data}))

