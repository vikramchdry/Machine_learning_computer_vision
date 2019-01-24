import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data',one_hot=True)

#input size is 784 tensor .Because our image size is 28*28 pixels

#num_classes is 10. we have 10 clasifier means 10 image categories

input_size = 784
num_classes = 10
batch_size = 50
total_batches = 1000
#x is input for our network.

x = tf.placeholder(tf.float32,[None,input_size])
y = tf.placeholder(tf.float32,[None,num_classes])

#y = mx+b we know for matrix multiplication is possible when 1st matrix column and 2nd matrix row is same

w = tf.Variable(tf.random_normal([input_size,num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

output = tf.matmul(x,w)+b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels= y))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for batch_no in range(total_batches):
    batch = data.train.next_batch(batch_size)
    train_images, train_labels = batch[0], batch[1]
    loss_value = sess.run([optimiser,loss],feed_dict={x:train_images,y:train_labels})
    print(loss_value)

predictions = tf.argmax(output, 1)
correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
test_images, test_labels = data.test.images, data.test.labels
accuracy_value = sess.run(accuracy_operation, feed_dict={x: test_images,
                                                            y: test_labels})

print('Accuracy : ', accuracy_value)
sess.close()

#Accuracy :  0.8019 for batch_size = 100

#Accuracy :  0.8799 for  total_batch = 1000,batch_size = 30

