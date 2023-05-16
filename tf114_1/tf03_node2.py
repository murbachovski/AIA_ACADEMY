import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

sess = tf.compat.v1.Session()

# 덧셈
node3 = tf.add(node1, node2)
print('덧셈')
print(sess.run(node3))

# 뺄셈
node4 = node1 - node2
print('뺄셈')
print(sess.run(node4))

# 곱셉
node5 = node1 * node2
print('곱셈')
print(sess.run(node5))

# 나눗셈
node6 = node1 / node2
print('나눗셈')
print(sess.run(node6))
