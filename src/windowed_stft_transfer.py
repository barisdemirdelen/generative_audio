import getopt
import random
import sys
import os

import librosa
import time
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import tensorflow as tf

import audio_utils
import params
from magenta.models.nsynth.utils import specgram, ispecgram
from stft_transfer import amplitude, StftTransfer

total_seconds = 600


def get_best_style(transferrer, current_initial, style_stft, hop_samples):
    best_style = None
    best_style_loss = float('inf')
    best_i = 0
    for i in range(random.randint(0, hop_samples), style_stft.shape[1], hop_samples):
        current_style = style_stft[:, i:i + current_initial.shape[1], :]
        if current_style.shape[1] != current_initial.shape[1]:
            continue
        loss = transferrer.get_style_loss(current_stft=current_initial, style_stft=current_style)
        # print(f'Sample {i}: {loss}')
        if loss < best_style_loss:
            best_style_loss = loss
            best_style = current_style
            best_i = i
    print(f'Best style at {best_i}')
    return best_style


def time_windowed_stft_transfer(content_file, style_file, window_seconds=5, hop_seconds=2.5):
    rate, content = wavfile.read(content_file)
    if total_seconds:
        content = content[:rate * total_seconds]
    total_samples = int(1.0 * params.sample_rate / rate * len(content))
    print('Resampling content')
    content = resample(content, total_samples)
    content = content.astype(np.float32)
    librosa.output.write_wav('output/content.wav', sr=params.sample_rate, y=content.astype(np.int16), norm=False)

    x = np.random.normal(0, 1, total_samples)
    x = x.astype(np.float32)
    x /= np.max(np.abs(x))
    x *= (amplitude - 1)
    # x *= 0.01
    x = content

    unnormal_noise = x.flatten()
    librosa.output.write_wav('output/x.wav', sr=params.sample_rate, y=unnormal_noise.astype(np.int16), norm=False)

    rate, style = wavfile.read(style_file)
    if total_seconds:
        style = style[:rate * total_seconds]
    total_samples2 = int(len(style) / rate) * params.sample_rate
    print('Resampling style')
    style = resample(style, total_samples2)
    style = style.astype(np.float32)
    librosa.output.write_wav('output/style.wav', sr=params.sample_rate, y=style.astype(np.int16), norm=False)
    session = tf.Session()

    style_stft = specgram(style, n_fft=params.fft_size, log_mag=audio_utils.log, hop_length=256, mag_only=True)
    style_stft = style_stft.reshape([style_stft.shape[0], style_stft.shape[1]])
    style_stft = style_stft.T
    style_stft = style_stft[np.newaxis, :, :]
    print('Style: ')
    print(f'Min: {np.min(style_stft):.4f}, Max: {np.max(style_stft):.4f}, Mean: {np.mean(style_stft):.4f}')

    content_stft = specgram(content, n_fft=params.fft_size, log_mag=audio_utils.log, hop_length=256, mag_only=True)
    content_stft = content_stft.reshape([content_stft.shape[0], content_stft.shape[1]])
    content_stft = content_stft.T
    content_stft = content_stft[np.newaxis, :, :]
    print('Content: ')
    print(f'Min: {np.min(content_stft):.4f}, Max: {np.max(content_stft):.4f}, Mean: {np.mean(content_stft):.4f}')

    result_stft = content_stft[:, :, :]

    hop_samples = int(content_stft.shape[1] / (len(content) / params.sample_rate) * hop_seconds)
    window_samples = int(content_stft.shape[1] / (len(content) / params.sample_rate) * window_seconds)

    transferrer = StftTransfer(session, (content_stft.shape[0], window_samples, content_stft.shape[2]))

    print('Starting window shifting')
    first = True
    for j in range(3):
        content_stft = result_stft[:]

        for i in range(0 if first else random.randint(0, hop_samples), result_stft.shape[1], hop_samples):
            start_time = time.time()
            # next_hop = int(hop_samples[i + 1]+1) if len(hop_samples) > i + 1 else content_stft.shape[1]
            current_content = content_stft[:, i:i + window_samples, :]
            current_initial = result_stft[:, i:i + window_samples, :]
            if current_content.shape[1] != window_samples:
                continue

            current_style = get_best_style(transferrer, current_initial[:], style_stft, hop_samples // 4)

            current_result_stft = transferrer.stft_transfer(content_stft=current_content, style_stft=current_style,
                                                            initial_stft=current_initial, maxiter=20)
            result_stft[:, i:i + window_samples, :] = current_result_stft
            print(
                f'Epoch {j} last sample: {(i/result_stft.shape[1]):.4f} elapsed time: {(time.time() - start_time):.4f}')

        result = result_stft[0].T[:, :, np.newaxis]
        result = ispecgram(result, params.fft_size, hop_length=256,
                           log_mag=audio_utils.log, mag_only=True, num_iters=20)
        result = np.clip(result, -1, 1)
        result *= amplitude

        # result = stft_transfer(content=content, style=style, initial=x)
        librosa.output.write_wav(f'output/result{j}.wav', sr=params.sample_rate, y=result.astype(np.int16), norm=False)
        print('Written result.wav')
        # librosa.output.write_wav('output/result.wav', sr=sample_rate, y=result.astype(np.int16), norm=False)
        # print('Written result.wav')
        first = False

    print('Inverting specgram')

    result = result_stft[0].T[:, :, np.newaxis]
    result = ispecgram(result, params.fft_size, hop_length=256,
                       log_mag=audio_utils.log, mag_only=True, num_iters=100)
    result = np.clip(result, -1, 1)
    result *= amplitude

    # result = stft_transfer(content=content, style=style, initial=x)
    librosa.output.write_wav('output/result.wav', sr=params.sample_rate, y=result.astype(np.int16), norm=False)
    print('Written result.wav')


if __name__ == '__main__':

    content_file = ''
    style_file = ''
    content_given = False
    style_given = False
    opts, args = getopt.getopt(sys.argv[1:], 'hc:s:', ['content=', 'style='])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('test.py -c <content_wav_file> -s <style_wav_file>')
            sys.exit()
        elif opt in ('-c', '--content'):
            content_file = arg
            content_given = True
        elif opt in ('-s', '--style'):
            style_file = arg
            style_given = True

    if style_given and not content_given:
        params.content_factor = 0.0

    if not os.path.isdir('output'):
        os.mkdir('output')

    time_windowed_stft_transfer(content_file, style_file, window_seconds=20, hop_seconds=10)
