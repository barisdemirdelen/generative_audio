import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.training.saver import Saver
import time

import params
from model.stft_classifier import StftClassifierModel

amplitude = 2 ** 14 + 1
continue_from = None


class StftTransfer(object):
    def __init__(self, session, stft_shape):
        self.start_time = 0
        self.model = None
        self.stft_input = None
        self.loss = None
        self.grad = None
        self.const_zero = None
        self.style_models = None
        self.content_features_graph = None
        self.content_features = None
        self.content_loss = None
        self.style_loss = None
        self.style_losses = None
        self.style_features_list = None
        self.style_stft = None
        self.content_stft = None
        self.initial_stft = None
        self.start_time = 0
        self.iteration_count = 0
        self.session = session
        self.stft_shape = stft_shape
        self.create_model()

    def stft_transfer(self, content_stft, style_stft, initial_stft, maxiter=130):
        self.start_time = 0
        self.iteration_count = 0
        self.style_stft = style_stft
        self.content_stft = content_stft
        self.initial_stft = initial_stft

        self.start_time = time.time()
        x0 = np.copy(initial_stft)
        x0 = x0.ravel()
        bounds = [[0.0, 1.0]]
        bounds = np.repeat(bounds, len(x0), axis=0)

        result = scipy.optimize.minimize(
            self.evaluate,
            x0,
            jac=True,
            tol=0.0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'disp': False,
                'gtol': 0.0,
                'ftol': 0.0,
                'maxfun': 100000,
                'maxiter': maxiter},
            callback=self.optimization_callback)

        stft_result = np.reshape(result.x, initial_stft.shape)[0]
        return stft_result

    def create_model(self):
        self.stft_input = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.stft_shape[1], self.stft_shape[2]])
        self.model = StftClassifierModel(1)
        self.model.inference(self.stft_input, training=False)
        self.session.run(tf.global_variables_initializer())
        self.style_models = [
            self.model.layers['x'],
            self.model.layers['conv1_1'],
            self.model.layers['conv1_2'],
            self.model.layers['conv1_3'],
            self.model.layers['conv1_4'],
            self.model.layers['conv_merge'],
        ]

        self.content_features_graph = self.model.layers['conv_merge']

        saver = Saver()
        # try:
        #     saver.restore(session, 'weights/stft_classifier/stft_classifier')
        # except NotFoundError:
        #     print('Not Found')

        style_loss = 0
        model_factors = [1.0] * len(self.style_models)
        sum_model_factor = np.sum(model_factors)
        style_losses = []
        self.style_features_list = []
        for style_model, model_factor in zip(self.style_models, model_factors):
            style_features = tf.placeholder(dtype=tf.float32, shape=[None, style_model.get_shape()[1].value,
                                                                     style_model.get_shape()[2].value])
            self.style_features_list.append(style_features)
            xs_gram = 0
            xs = style_features[0]
            xs = tf.reshape(xs, [-1, xs.get_shape()[-1].value])
            xs_gram += 1.0 * tf.matmul(tf.transpose(xs), xs) / xs.get_shape()[-1].value

            xg = style_model[0]
            xg = tf.reshape(xg, [-1, xg.get_shape()[-1].value])
            xg_gram = 1.0 * tf.matmul(tf.transpose(xg), xg) / xs.get_shape()[-1].value
            normalizer = tf.reduce_mean(tf.square(xs_gram)) * xs.get_shape()[0].value ** 2
            style_loss = model_factor * tf.nn.l2_loss(xs_gram - xg_gram) / normalizer
            style_losses.append(params.style_factor * style_loss)

        self.style_losses = tf.stack(style_losses)
        self.style_loss = tf.reduce_mean(style_losses)
        self.const_zero = tf.constant(0)

        self.content_features = tf.placeholder(dtype=tf.float32, shape=[None, self.content_features_graph.shape[1],
                                                                        self.content_features_graph.shape[2]])
        xc = self.content_features[0]
        xg = self.content_features_graph[0]
        self.content_loss = params.content_factor * tf.nn.l2_loss(xg - xc) / tf.reduce_mean(xc ** 2) / xc.get_shape()[0].value

        self.loss = 0
        if params.content_factor > 0:
            self.loss += self.content_loss
        if params.style_factor > 0:
            self.loss += self.style_loss
        self.loss *= 1e4
        self.grad = tf.gradients(self.loss, [self.stft_input])
        self.grad = tf.reshape(self.grad, [-1])

    def evaluate(self, xn):
        current_xn = np.reshape(xn, self.initial_stft.shape)

        run_list = [self.content_loss if params.content_factor > 0 else self.const_zero,
                    self.style_loss if params.style_factor > 0 else self.const_zero,
                    self.const_zero,
                    self.const_zero,
                    self.loss, self.grad,
                    self.style_losses if params.style_factor > 0 else [self.const_zero]]

        style_feed_list = []
        for style_model in self.style_models:
            style_features = self.session.run(style_model, feed_dict={self.stft_input: self.style_stft})
            style_feed_list.append(style_features)
        content_feed = self.session.run(self.content_features_graph, feed_dict={self.stft_input: self.content_stft})

        current_feed_dict = {
            self.stft_input: current_xn,
            self.content_features: content_feed,
        }
        current_feed_dict.update({feature: feed for feature, feed in zip(self.style_features_list, style_feed_list)})

        current_content_loss, current_style_loss, current_loudness_loss, current_zero_loss, total_loss, gradients, current_style_losses = self.session.run(
            run_list,
            feed_dict=current_feed_dict)

        return total_loss.astype(np.float64), gradients.astype(np.float64)

    def optimization_callback(self, xn, force=False):
        self.iteration_count += 1

    def get_style_loss(self, current_stft, style_stft):
        style_feed_list = []
        for style_model in self.style_models:
            style_features = self.session.run(style_model, feed_dict={self.stft_input: style_stft})
            style_feed_list.append(style_features)

        current_feed_dict = {
            self.stft_input: current_stft,
        }
        current_feed_dict.update({feature: feed for feature, feed in zip(self.style_features_list, style_feed_list)})

        current_style_loss = self.session.run(self.style_loss, feed_dict=current_feed_dict)
        return current_style_loss
