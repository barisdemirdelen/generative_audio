import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import elu


def conv2d_layer(x, filter, units, name, strides=None, padding='SAME'):
    if strides is None:
        strides = [1, 1]
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(f'{name}/weights', shape=[filter[0], filter[1], x.get_shape()[-1].value, units],
                                 initializer=tf.random_normal_initializer(0, 1e-2))
        conv = tf.nn.conv2d(x, filter=kernel, strides=[1, strides[0], strides[1], 1], padding=padding)
        # biases = tf.get_variable('%s/biases' % name, shape=[units],
        #                          initializer=tf.constant_initializer(0.0))
        # out = tf.nn.bias_add(conv, biases)
        post_out = leaky_relu(conv, name=name)
        # post_out = out
        tf.summary.histogram(f'{name}/kernel', kernel)
        # tf.summary.histogram('%s/biases' % name, biases)
    return post_out


def leaky_relu(x, alpha=0.01, name=None):
    return tf.maximum(alpha * x, x, name=name)


def atrous_conv1d(value, filters, rate, padding):
    value_2d = tf.expand_dims(value, 2)
    filters_2d = tf.expand_dims(filters, 1)
    return tf.squeeze(tf.nn.atrous_conv2d(value_2d, filters_2d, rate, padding), [2])


# Dilation goes through 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, ...
def gated_unit(x, w1, w2, dilation):
    # x must be [batch, width, channel]
    # w1, w2 must be [filter_width, in_channels, out_channels]
    dilated1 = atrous_conv1d(x, w1, dilation, 'VALID')
    # dilated2 = atrous_conv1d(x, w2, dilation, 'VALID')
    z = tf.tanh(dilated1)
    # one_by_one = tf.nn.conv1d(z, cw, 1, 'VALID')
    return z


def gated_conv1d_layer(x, kernel, num_filters, name, dilation=1, stride=1, activation=True, padding='VALID'):
    with tf.name_scope(name) as scope:
        kernel1 = tf.get_variable(f'{name}/weights1', shape=[kernel, x.get_shape()[-1].value, num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=1e-3))
        # kernel2 = tf.get_variable(f'{name}/weights2', shape=[kernel, x.get_shape()[-1].value, num_filters],
        #                           initializer=tf.truncated_normal_initializer(stddev=1e-5))
        post_out = gated_unit(x, kernel1, None, dilation)
        # biases = tf.get_variable(f'{name}/biases', shape=[num_filters],
        #                          initializer=tf.constant_initializer(0.0))
        # post_out = tf.nn.bias_add(post_out, biases)
        # if activation:
        #     post_out = leaky_relu(post_out, name=name)

        tf.summary.histogram(f'{name}/kernel', kernel)
        # tf.summary.histogram(f'{name}/biases', biases)
    return post_out


def dilated_conv1d_layer(x, kernel, num_filters, name, dilation=1, stride=1, activation=True, padding='VALID'):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(f'{name}/weights', shape=[kernel, x.get_shape()[-1].value, num_filters],
                                 initializer=tf.truncated_normal_initializer(stddev=1e-2))
        post_out = atrous_conv1d(x, kernel, dilation, padding=padding)
        biases = tf.get_variable(f'{name}/biases', shape=[num_filters],
                                 initializer=tf.constant_initializer(0.0))
        post_out = tf.nn.bias_add(post_out, biases)
        if activation:
            post_out = elu(post_out, name=name)

        # tf.summary.histogram(f'{name}/kernel', kernel)
        # tf.summary.histogram(f'{name}/biases', biases)
    return post_out


def conv1d_layer(x, kernel, num_filters, name, stride=1, activation=True, padding='VALID'):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(f'{name}/weights', shape=[kernel, x.get_shape()[-1].value, num_filters],
                                 initializer=tf.truncated_normal_initializer(stddev=1e-5))
        post_out = tf.nn.conv1d(x, kernel, stride=stride, padding=padding)
        # biases = tf.get_variable(f'{name}/biases', shape=[num_filters],
        #                          initializer=tf.constant_initializer(0.0))
        # post_out = tf.nn.bias_add(post_out, biases)
        if activation:
            post_out = leaky_relu(post_out, name=name)

        tf.summary.histogram(f'{name}/kernel', kernel)
        # tf.summary.histogram(f'{name}/biases', biases)
    return post_out


def conv1d_transpose_layer(x, kernel, output_shape, name, activation=True):
    with tf.variable_scope(name) as scope:
        # h, w, out, in
        output_shape.insert(1, 1)
        kernel = tf.get_variable(f'{name}/weights', [1, kernel, output_shape[-1], x.get_shape()[-1].value],
                                 initializer=tf.truncated_normal_initializer(stddev=1e-5))
        biases = tf.get_variable(f'{name}/biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(x, [-1, 1, x.get_shape()[1].value, x.get_shape()[2].value])
        convt = tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, 1, 2, 1])
        convt = tf.reshape(convt, [-1, convt.get_shape()[2].value, convt.get_shape()[3].value])
        post_out = tf.nn.bias_add(convt, biases)
        if activation:
            post_out = leaky_relu(post_out, name=name)

        tf.summary.histogram(f'{name}/kernel', kernel)
        tf.summary.histogram(f'{name}/biases', biases)

        return post_out


def flatten2d(x, name):
    with tf.name_scope(name) as scope:
        flatten_out = tf.reshape(x, [-1, x.get_shape()[1].value * x.get_shape()[2].value * x.get_shape()[3].value],
                                 name=name)
    return flatten_out


def flatten1d(x, name):
    with tf.name_scope(name) as scope:
        flatten_out = tf.reshape(x, [-1, x.get_shape()[1].value * x.get_shape()[2].value],
                                 name=name)
    return flatten_out


def fully_connected(x, units, name, activation=True):
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(f'{name}/weights', shape=[x.get_shape()[-1].value, units],
                                  initializer=tf.random_normal_initializer(0, 1e-2))
        biases = tf.get_variable(f'{name}/biases', shape=[units],
                                 initializer=tf.constant_initializer(0.0))
        out = tf.matmul(x, weights)
        fc = tf.nn.bias_add(out, biases)
        if activation:
            fc = tf.nn.elu(out, name=scope)
        #
        # tf.summary.histogram(f'{name}/kernel', weights)
        # tf.summary.histogram(f'{name}/biases', biases)

        return fc


def pool1d(x, name):
    x = tf.reshape(x, [-1, 1, x.get_shape()[1].value, x.get_shape()[2].value])
    pool = tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME', name=f'{name}/pool')
    pool = tf.reshape(pool, [-1, pool.get_shape()[2].value, pool.get_shape()[3].value])
    return pool


def pool2d(x, name):
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=f'{name}/pool')
    return pool
