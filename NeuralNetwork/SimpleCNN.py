"""
Excerpt from Tensorflow 实战
"""

from Util.BatchMaker import BatchMaker
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


# routines
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    :param x:
    :param W: [n_rows, n_cols, n_layers, n_kernels]
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # strides is the offset of window in each step


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # here ksize and strides are in [batch, height, width, channels]


# main function
def run(train_batch_feeder, test_batch_feeder):
    assert isinstance(train_batch_feeder, BatchMaker)
    assert isinstance(test_batch_feeder, BatchMaker)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # the 1st convolution layer
    W_conv1 = weight_variable([5, 5, 1, 32])    # 32 conv kernel, each is 5rows * 5cols * 1channel
    b_conv1 = bias_variable([32])
    # the virtue of ReLu activation function is its not being susceptible to gradient diffusion
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)        # relu(f) === max(f, 0)
    h_pool1 = max_pool_2x2(h_conv1)

    # the 2nd convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])   # full connection between two conv layers
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # now the shape of tensor is 64kernels * [7*7]image

    # reshape and full connection layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])             # 1024 is arbitrary
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])    # flatten for kx+b
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout Layer
    keep_prob = tf.placeholder(tf.float32)                  # threshold passed at runtime
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)            # drop some to avoid over-fitting

    # Softmax layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # evaluation function
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start
    tf.global_variables_initializer().run()
    for i in range(8000):
        # batch = mnist.train.next_batch(50)
        batch = train_batch_feeder.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("setp %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    batch = test_batch_feeder.all()
    print("test accuracy %g" % accuracy.eval(feed_dict={
        # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
        x: batch[0], y_: batch[1], keep_prob: 1.0
    }))


if __name__ == '__main__':
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.InteractiveSession()

    train_b = BatchMaker(x=np.random.rand(10, 1), y_=np.random.rand(10, 1))
    test_b = BatchMaker(x=np.random.rand(10, 1), y_=np.random.rand(10, 1))
    run(train_b, test_b)