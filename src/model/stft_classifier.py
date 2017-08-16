import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.layers.normalization import batch_normalization

from model.ops import dilated_conv1d_layer, flatten1d, fully_connected


class StftClassifierModel(object):
    def __init__(self, batch_size):
        self.weight_regularizer = l2_regularizer(1e-3)
        self.layers = {}
        self.reuse = False
        self.batch_size = batch_size

    def inference(self, x, training=True):
        logits = x
        with tf.variable_scope("mfcc", self.reuse):
            self.reuse = True
            kernel = 3
            filter_factor = 8

            x = batch_normalization(x)

            conv1_1 = dilated_conv1d_layer(x, kernel=kernel, num_filters=128 * filter_factor, stride=1, dilation=1,
                                           name="conv1_1",
                                           padding="SAME")
            conv1_2 = dilated_conv1d_layer(x, kernel=kernel, num_filters=64 * filter_factor, stride=1, dilation=2,
                                           name="conv1_2",
                                           padding="SAME")
            conv1_3 = dilated_conv1d_layer(x, kernel=kernel, num_filters=32 * filter_factor, stride=1, dilation=4,
                                           name="conv1_3",
                                           padding="SAME")
            conv1_4 = dilated_conv1d_layer(x, kernel=kernel, num_filters=16 * filter_factor, name="conv1_4", stride=1,
                                           dilation=8,
                                           padding="SAME")
            # conv1_5 = dilated_conv1d_layer(x, kernel=kernel, num_filters=8 * filter_factor, name="conv1_5", stride=1,
            #                                dilation=16,
            #                                padding="SAME")
            # conv1_6 = dilated_conv1d_layer(x, kernel=kernel, num_filters=4 * filter_factor, name="conv1_6", stride=1,
            #                                dilation=32,
            #                                padding="SAME")
            # conv1_7 = dilated_conv1d_layer(x, kernel=kernel, num_filters=2 * filter_factor, name="conv1_7", stride=1,
            #                                dilation=64,
            #                                padding="SAME")
            # conv1_8 = dilated_conv1d_layer(x, kernel=kernel, num_filters=2 * filter_factor, name="conv1_8", stride=1,
            #                                dilation=128,
            #                                padding="SAME")
            #
            # conv1_merge = tf.concat((conv1_1, conv1_2, conv1_3, conv1_4, conv1_5, conv1_6, conv1_7, conv1_8), axis=2)
            conv1_merge = tf.concat((conv1_1, conv1_2, conv1_3, conv1_4), axis=2)

            conv2_1 = dilated_conv1d_layer(conv1_1, kernel=kernel, num_filters=1024, name="conv2_1", stride=1,
                                           dilation=2,
                                           padding="SAME")
            conv3_1 = dilated_conv1d_layer(conv2_1, kernel=kernel, num_filters=1024, name="conv3_1", stride=1,
                                           dilation=2,
                                           padding="SAME")
            conv4_1 = dilated_conv1d_layer(conv3_1, kernel=kernel, num_filters=16, name="conv4_1", stride=1,
                                           dilation=2,
                                           padding="SAME")

            self.layers["x"] = x
            self.layers["conv1_1"] = conv1_1
            self.layers["conv1_2"] = tf.reshape(conv1_2, [-1, x.get_shape()[1].value, conv1_2.get_shape()[2].value],
                                                name=None)
            self.layers["conv1_3"] = tf.reshape(conv1_3, [-1, x.get_shape()[1].value, conv1_3.get_shape()[2].value],
                                                name=None)
            self.layers["conv1_4"] = tf.reshape(conv1_4, [-1, x.get_shape()[1].value, conv1_4.get_shape()[2].value],
                                                name=None)
            self.layers["conv_merge"] = tf.reshape(conv1_merge, [-1, x.get_shape()[1].value, conv1_merge.get_shape()[2].value],
                                                   name=None)
            self.layers["conv3_1"] = conv3_1

            if training:
                pre_flatten = tf.reshape(conv4_1, [-1, x.get_shape()[1].value, conv4_1.get_shape()[2].value],
                                         name='pre_flatten')
                flatten = flatten1d(pre_flatten, name='flatten')
                # fc1 = fully_connected(flatten, units=512, name='fc1')
                # fc2 = fully_connected(fc1, units=256, name='fc2')

                logits = fully_connected(flatten, units=26, name='logits', activation=False)

        return logits

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            sigmoid_qualities = tf.nn.sigmoid(logits[:, 2:12])
            correct_prediction_qualities = tf.equal(tf.round(sigmoid_qualities), tf.round(labels[:, 2:12]))
            accuracy_qualities = tf.reduce_mean(tf.cast(correct_prediction_qualities, tf.float32)) * 10

            softmax_families = tf.nn.softmax(logits[:, 12:23])
            correct_prediction_families = tf.equal(tf.argmax(softmax_families, 1), tf.argmax(labels[:, 12:23], 1))
            accuracy_families = tf.reduce_mean(tf.cast(correct_prediction_families, tf.float32)) * 11

            softmax_sources = tf.nn.softmax(logits[:, 23:])
            correct_prediction_sources = tf.equal(tf.argmax(softmax_sources, 1), tf.argmax(labels[:, 23:], 1))
            accuracy_sources = tf.reduce_mean(tf.cast(correct_prediction_sources, tf.float32)) * 3

            accuracy = (accuracy_qualities + accuracy_families + accuracy_sources) / 24

        return accuracy

    def loss(self, logits, labels, training):
        cross_entropy_pitch_velocity = 2 * tf.reduce_mean((tf.sigmoid(logits[:, 2]) - labels[:, 2]) ** 2)
        cross_entropy_qualities = 10 * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[:, 2:12], labels=labels[:, 2:12]))
        cross_entropy_families = 11 * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, 12:23], labels=labels[:, 12:23]))
        cross_entropy_sources = 3 * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, 23:], labels=labels[:, 23:]))
        cross_entropy = cross_entropy_pitch_velocity + cross_entropy_qualities + cross_entropy_families + cross_entropy_sources
        cross_entropy /= 26
        regularizer = 0
        if self.weight_regularizer is not None:
            for parameter in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                regularizer += self.weight_regularizer(parameter)
        # loss = cross_entropy + regularizer
        loss = tf.cond(training, lambda: cross_entropy + regularizer, lambda: cross_entropy)
        # tf.summary.scalar("cross_entropy_loss", cross_entropy)
        # tf.summary.scalar("regularization_loss", regularizer)
        # tf.summary.scalar("total_loss", loss)

        return loss
