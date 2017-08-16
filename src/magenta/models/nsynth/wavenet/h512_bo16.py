# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A WaveNet-style AutoEncoder Configuration."""

# internal imports
import tensorflow as tf

from magenta.models.nsynth import reader
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import masked


class Config(object):
  """Configuration object that helps manage the graph."""

  def __init__(self, train_path=None):
    self.num_iters = 200000
    self.learning_rate_schedule = {
        0: 2e-4,
        90000: 4e-4 / 3,
        120000: 6e-5,
        150000: 4e-5,
        180000: 2e-5,
        210000: 6e-6,
        240000: 2e-6,
    }
    self.ae_hop_length = 512
    self.ae_bottleneck_width = 16
    self.train_path = train_path
    self.layers = []

  def get_batch(self, batch_size):
    assert self.train_path is not None
    data_train = reader.NSynthDataset(self.train_path, is_training=True)
    return data_train.get_wavenet_batch(batch_size, length=6144)

  @staticmethod
  def _condition(x, encoding):
    """Condition the input on the encoding.

    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.

    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """
    mb, length, channels = x.get_shape().as_list()
    enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
    assert enc_mb == mb
    assert enc_channels == channels

    encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
    x = tf.reshape(x, [mb, enc_length, -1, channels])
    x += encoding
    x = tf.reshape(x, [mb, length, channels])
    x.set_shape([mb, length, channels])
    return x

  def build(self, inputs, is_training):
    """Build the graph for this configuration.

    Args:
      inputs: A dict of inputs. For training, should contain 'wav'.
      is_training: Whether we are training or not. Not used in this config.

    Returns:
      A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
      the 'quantized_input', and whatever metrics we want to track for eval.
    """
    del is_training
    num_stages = 10
    num_layers = 30
    filter_length = 3
    width = 512
    skip_width = 256
    ae_num_stages = 10
    ae_num_layers = 30
    ae_filter_length = 3
    ae_width = 128

    # Encode the source with 8-bit Mu-Law.
    x = inputs
    x_quantized = x
    # x_quantized = x
    x_scaled = x_quantized / 128.0
    x_scaled = tf.expand_dims(x_scaled, 2)

    ###
    # The Non-Causal Temporal Encoder.
    ###
    en = masked.conv1d(
        x_scaled,
        causal=False,
        num_filters=ae_width,
        filter_length=ae_filter_length,
        name='ae_startconv')
    # self.layers.append(en)

    for num_layer in range(ae_num_layers):
      dilation = 2**(num_layer % ae_num_stages)
      d = tf.nn.relu(en)
      d = masked.conv1d(
          d,
          causal=False,
          num_filters=ae_width,
          filter_length=ae_filter_length,
          dilation=dilation,
          name='ae_dilatedconv_%d' % (num_layer + 1))
      d = tf.nn.relu(d)

      en += masked.conv1d(
          d,
          num_filters=ae_width,
          filter_length=1,
          name='ae_res_%d' % (num_layer + 1))

    en = masked.conv1d(
        en,
        num_filters=self.ae_bottleneck_width,
        filter_length=1,
        name='ae_bottleneck')
    en = masked.pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg')
    encoding = en
    # self.layers.append(en)



    ###
    # The WaveNet Decoder.
    ###
    l = masked.shift_right(x_scaled)
    l = masked.conv1d(
        l, num_filters=width, filter_length=filter_length, name='startconv')

    # Set up skip connections.
    s = masked.conv1d(
        l, num_filters=skip_width, filter_length=1, name='skip_start')

    # Residual blocks with skip connections.
    for i in range(num_layers):
      dilation = 2**(i % num_stages)
      d = masked.conv1d(
          l,
          num_filters=2 * width,
          filter_length=filter_length,
          dilation=dilation,
          name='dilatedconv_%d' % (i + 1))
      d = self._condition(d,
                          masked.conv1d(
                              en,
                              num_filters=2 * width,
                              filter_length=1,
                              name='cond_map_%d' % (i + 1)))

      assert d.get_shape().as_list()[2] % 2 == 0
      m = d.get_shape().as_list()[2] // 2
      d_sigmoid = tf.sigmoid(d[:, :, :m])
      d_tanh = tf.tanh(d[:, :, m:])
      d = d_sigmoid * d_tanh

      self.layers.append(d)

      l += masked.conv1d(
          d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
      s += masked.conv1d(
          d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
    s = self._condition(s,
                        masked.conv1d(
                            en,
                            num_filters=skip_width,
                            filter_length=1,
                            name='cond_map_out1'))
    s = tf.nn.relu(s)

    ###
    # Compute the logits and get the loss.
    ###
    logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
    logits = tf.reshape(logits, [-1, 256])
    probs = tf.nn.softmax(logits, name='softmax')
    x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=x_indices, name='nll'),
        0,
        name='loss')

    return {
        'predictions': probs,
        'loss': loss,
        'eval': {
            'nll': loss
        },
        'quantized_input': x_quantized,
        'encoding': encoding,
    }
