import tensorflow as tf

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
z = tf.add(x, y, name='sum')
sess = tf.Session()
summary_writer = tf.summary.FileWriter('/tmp/1',sess.graph)
summary_writer.flush()