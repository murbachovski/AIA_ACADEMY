import tensorflow as tf
tf.set_random_seed(1253)

# 1. DATA
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. MODEL
# y = wx + b 
hypothesis = x * w + b

# 3-1. COMPILE
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-2 COMPILE
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Variable 초기화

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:[6,7,8], y:[14, 5, 6]})
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)


#############################실습######################################
        x_data = [6,7,8]
        # 예측값을 뽑아라~~~12,14,16

        x_test = tf.placeholder(tf.float32, shape=[None])
        
        y_predict = x_data * w_val + b_val
        
        print('[6,7,8]예측: ', sess.run(y_predict, feed_dict={x_test:x_data}))