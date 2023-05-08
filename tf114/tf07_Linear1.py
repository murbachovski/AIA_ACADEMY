import tensorflow as tf
tf.set_random_seed(1253)


# 1. DATA
x = [1, 2, 3]
y = [1, 2, 3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(100, dtype=tf.float32) # bias = 통상 0입니다.


# 2. MODEL
# y = wx + b 
hypothesis = x * w + b

# 3-1. COMPILE
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

# 통으로 외우래
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법 방식으로 optimizer를 최적화시켜서 loss값을 뽑는다.
train = optimizer.minimize(loss)
# 통으로 외우래
# 한 줄로 요약하면 model.compile(loss='mse', optimizer='sgd')    # gradient descent

# 3-2 COMPILE
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # Variable 초기화

epochs = 2000
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))

sess.close
