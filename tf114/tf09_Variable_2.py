import tensorflow as tf
tf.compat.v1.set_random_seed(123)

### [실습]

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)

# 3-2 COMPILE
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Variable 초기화

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:[1,2,3,4,5], y:[2,4,6,8,10]})
        if step %99 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)

#############################실습######################################
    x_data = [6,7,8]

    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    
    변수 = x_test * w_val + b_val
    
    print('[6,7,8]예측: ', sess.run(변수, feed_dict={x_test:x_data}))

    변수 = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')
    # print(변수)
    # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

    # 초기화 첫번째
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    aaa = sess.run(변수)
    print('aaa:', aaa)  # aaa: [0.86205626]
    sess.close()

    # 초기화 두번째
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    bbb = 변수.eval(session=sess)   # .eval로 변수 변경이 가능하다. # 텐서플로 데이터형인 '변수'를 파이썬에서 볼 수 있는 놈으로  바꿔줘.
    print('bbb:', bbb)  # bbb: [0.86205626]
    sess.close()

    # 초기화 세번째
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    ccc = 변수.eval()
    print('ccc:', ccc)  # ccc: [0.86205626]
    sess.close()

    