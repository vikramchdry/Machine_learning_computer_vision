import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data',one_hot=True)

#input size is 784 tensor .Because our image size is 28*28 pixels

#num_classes is 10. we have 10 clasifier means 10 image categories

input_size = 784
num_classes = 10
batch_size = 50
total_batches = 1000

x = tf.placeholder(tf.float32,[None,input_size])
y = tf.placeholder(tf.float32,[None,num_classes])


def add_variable(tf_variable,summary_name):
    with tf.name_scope(summary_name + '_summary'):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar('Mean',mean)
        with tf.name_scope('standard_dev'):
            standard_dev = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
        tf.summary.scalar('StandardDeviation', standard_dev)
        tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
        tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
        tf.summary.histogram('Histogram', tf_variable)

x_input_reshape = tf.reshape(x, [-1, 28, 28, 1],
                             name='input_reshape')

def conv_net(input_layer,filter , kernel_size = [3,3],
             activation = tf.nn.relu):
    layer = tf.layers.conv2d(inputs=input_layer,filters=filter,kernel_size=kernel_size,activation=activation)
    add_variable = (layer,'convolution')
    return layer
def pooling_layer(input_layer,pool_size = [2,2],strides = 2):
    layer = tf.layers.max_pooling2d(inputs=input_layer,pool_size = pool_size,strides=strides)
    add_variable(layer,'pooling')
    return layer

def dense_layer(input_layer, units, activation=tf.nn.relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation
    )
    add_variable(layer, 'dense')
    return layer

con1 = conv_net(x_input_reshape,64)
pooling_layer_1 = pooling_layer(con1)
con2 = conv_net(pooling_layer_1,128)
pooling_layer_2 = pooling_layer(con2)
flattened_pool = tf.reshape(pooling_layer_2, [-1, 5 * 5 * 128],
                            name='flattened_pool')
dense_layer_bottleneck = dense_layer(flattened_pool,1024)
dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
        inputs=dense_layer_bottleneck,
        rate=0.4,
        training=dropout_bool
    )
logits = dense_layer(dropout_layer, num_classes)
with tf.name_scope('loss'):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(

        labels=y, logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)


with tf.name_scope('optimiser'):
    optimiser = tf.train.AdamOptimizer().minimize(loss_operation)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32))
tf.summary.scalar('accuracy', accuracy_operation)

session = tf.Session()
session.run(tf.global_variables_initializer())

merged_summary_operation = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(r'C:\Users\vikram singh\PycharmProjects\DeepMind\Training_model', session.graph)
test_summary_writer = tf.summary.FileWriter(r'C:\Users\vikram singh\PycharmProjects\DeepMind\testing')

test_images, test_labels = data.test.images, data.test.labels

for batch_no in range(total_batches):
    mnist_batch = data.train.next_batch(batch_size)
    train_images, train_labels = mnist_batch[0], mnist_batch[1]
    _, merged_summary = session.run([optimiser, merged_summary_operation],
                                    feed_dict={
        x: train_images,
        y: train_labels,
        dropout_bool: True
    })
    train_summary_writer.add_summary(merged_summary, batch_no)
    if batch_no % 10 == 0:
        merged_summary, _ = session.run([merged_summary_operation,
                                         accuracy_operation], feed_dict={
            x: test_images,
            y: test_labels,
            dropout_bool: False
        })
        test_summary_writer.add_summary(merged_summary, batch_no)