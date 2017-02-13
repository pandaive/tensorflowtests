import tensorflow as tf
save_file = "./model_0.ckpt"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True)

learning_rate = 0.001
n_input = 784
n_classes = 10

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.truncated_normal([n_input,n_classes]))
bias = tf.Variable(tf.truncated_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

batch_size = 128
n_epochs = 100

with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(accuracy, feed_dict = {features: mnist.test.images, labels: mnist.test.labels})

    print('Test Accuracy: {}'.format(test_accuracy))