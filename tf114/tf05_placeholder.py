import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())

# 즉시실행모드
# tf.compat.v1.disable_eager_execution() Off

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4.5}))
