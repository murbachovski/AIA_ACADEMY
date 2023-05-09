import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)

# 3-2 COMPILE

loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Variable 초기화

    epochs = 201
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)
        w_val_list.append(w_val)

#############################실습######################################
    x_data = [6,7,8]

    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    
    y_predict = x_test * w_val + b_val
    
    print('[6,7,8]예측: ', sess.run(y_predict, feed_dict={x_test:x_data}))

print(loss_val_list)
print(w_val_list)

# plt.plot(loss_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

# plt.scatter(w_val_list, loss_val_list)
# plt.xlabel('weight')
# plt.ylabel('loss')
# plt.show()