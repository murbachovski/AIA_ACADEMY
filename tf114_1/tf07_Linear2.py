import tensorflow as tf
tf.set_random_seed(337)

# 1. DATA
x = [1,2,3,4,5]
y = [2,4,6,8,10]

w = tf.Variable(555, dtype=tf.float32)
b = tf.Variable(300, dtype=tf.float32)


#####[실습] MAKE IT#####
# w = 2, b = 2

# 2. MODEL
# y = wx + b 
hypothesis = x * w + b

# 3-1. COMPILE
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

# 통으로 외우래
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03) # 경사하강법 방식으로 optimizer를 최적화시켜서 loss값을 뽑는다.
train = optimizer.minimize(loss)
# 통으로 외우래
# 한 줄로 요약하면 model.compile(loss='mse', optimizer='sgd')    # gradient descent

# 3-2 COMPILE
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # Variable 초기화

epochs = 2000
for step in range(epochs):
    sess.run(train)
    if step %200 == 0:
        print(step, sess.run(loss), round(sess.run(w)), (sess.run(b)))

sess.close
# 1800 5.570655e-13 2.0 1.8935822e-06