import tensorflow as tf

k_ouput = 64

image_width = 10
image_height = 10
color_channels = 3

filter_size_width = 5
filter_size_height = 5

input = tf.placeholder(tf.float32, shape = [None, image_width, image_height, color_channels])

weight = tf.Variable(tf.truncated_normal([filter_size_width, filter_size_height, color_channels, k_ouput]))
bias = tf.Variable(tf.zeros(k_ouput))

conv_layer = tf.nn.conv2d(input, weight, strides = [1, 2, 2, 1], padding = 'SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)

conv_layer = tf.nn.max_pool(conv_layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

