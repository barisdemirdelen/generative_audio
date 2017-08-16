import librosa
import numpy as np
import copy

from numpy.lib.stride_tricks import as_strided
from scipy.fftpack import dct, idct
from scipy.signal import butter, lfilter
import scipy.ndimage
import tensorflow as tf

log = False


# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError('Window size must be even!')
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def halfoverlap(X, window_size):
    """
    Create an overlapped version of X using 50% of window_size as overlap.
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError('Window size must be even!')
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def invert_halfoverlap(X_strided):
    """
    Invert ``halfoverlap`` function to reconstruct X
    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows
    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    # Hardcoded 50% overlap! Can generalize later...
    n_rows, n_cols = X_strided.shape
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),)).astype(X_strided.dtype)
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


def denoise(spectogram):
    denoised = np.copy(spectogram)
    print(np.mean(denoised))
    print(np.min(denoised))
    denoised = np.log1p(denoised)
    print(np.mean(denoised))
    print(np.min(denoised))
    denoised[np.where(denoised < 8)] = 0
    denoised = np.expm1(denoised)
    print(np.mean(denoised))
    print(np.min(denoised))
    return denoised


def revert_stft(y, fft_size, num_iter):
    if log:
        y = np.expm1(y)
    p = 2 * np.pi * np.random.random_sample(y.shape) - np.pi
    x = None
    for i in range(num_iter):
        S = y * np.exp(1j * p)
        x = librosa.istft(S, hop_length=fft_size // 4)
        p = np.angle(librosa.stft(x, n_fft=fft_size, hop_length=fft_size // 4))
    return x


def stft(x, fft_size):
    S = librosa.stft(x, n_fft=fft_size, hop_length=fft_size // 4)
    S = np.abs(S)
    if log:
        S = np.log1p(S)
    return S


def istft(X, fftsize=128, mean_normalize=True, real=False,
          compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = np.fft.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = np.fft.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2] = X
        X_pad[:, fftsize // 2:] = 0
        X = X_pad
    X = local_ifft(X).astype('float64')
    X = invert_halfoverlap(X)
    if mean_normalize:
        X -= np.mean(X)
    return X


def pretty_spectrogram(d, log=True, thresh=5, fft_size=512, step_size=64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False,
                           compute_onesided=True))

    maxi = 1
    if log == True:
        maxi = specgram.max()
        specgram /= maxi  # volume normalize to max 1
        # print('Max :' + str(maxi))
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    return specgram, maxi


# Also mostly modified or taken from https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
def invert_pretty_spectrogram(X_s, log=True, fft_size=512, step_size=512 / 4, n_iter=10):
    if log == True:
        X_s = np.power(10, X_s)

    X_s = np.concatenate([X_s, X_s[:, ::-1]], axis=1)
    X_t = iterate_invert_spectrogram(X_s, fft_size, step_size, n_iter=n_iter)
    return X_t


def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    for i in range(n_iter):
        if verbose:
            print(f'Runnning iter {i}')
        if i == 0:
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=True)
        else:
            # Calculate offset was False in the MATLAB version
            # but in mine it massively improves the result
            # Possible bug in my impl?
            X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                     set_zero_phase=False)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        phase = est / np.maximum(reg, np.abs(est))
        X_best = X_s * phase[:len(X_s)]
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)


def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print('WARNING: Large step size >50\% detected! '
                      'This code works best with high overlap - try '
                      'with 75% or greater')
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
                                           est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave


def xcorr_offset(x1, x2):
    """
    Under MSR-LA License
    Based on MATLAB implementation from Spectrogram Inversion Toolbox
    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.
    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.
    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype(np.float64), x2[::-1].astype(np.float64))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset


def freq_to_mel(f):
    return 2595. * np.log10(1 + (f / 700.))


def mel_to_freq(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def create_mel_filter(fft_size, n_freq_components=64, start_freq=300, end_freq=8000, rate=44100):
    """
    Creates a filter to convolve with the spectrogram to get out mels

    """
    spec_size = fft_size / 2
    start_mel = freq_to_mel(start_freq)
    end_mel = freq_to_mel(end_freq)
    plt_spacing = []
    # find our central channels from the spectrogram
    for i in range(10000):
        y = np.linspace(start_mel, end_mel, num=i, endpoint=False)
        logp = mel_to_freq(y)
        logp = logp / (rate / 2 / spec_size)
        true_spacing = [int(i) - 1 for i in np.ceil(logp)]
        plt_spacing_mel = np.unique(true_spacing)
        if len(plt_spacing_mel) == n_freq_components:
            break
    plt_spacing = plt_spacing_mel
    if plt_spacing_mel[-1] == spec_size:
        plt_spacing_mel[-1] = plt_spacing_mel[-1] - 1
    # make the filter
    mel_filter = np.zeros((int(spec_size), n_freq_components))
    # Create Filter
    for i in range(len(plt_spacing)):
        if i > 0:
            if plt_spacing[i - 1] < plt_spacing[i] - 1:
                # the first half of the window should start with zero
                mel_filter[plt_spacing[i - 1]:plt_spacing[i], i] = np.arange(0, 1,
                                                                             1. / (plt_spacing[i] - plt_spacing[i - 1]))
        if i < n_freq_components - 1:
            if plt_spacing[i + 1] > plt_spacing[i] + 1:
                mel_filter[plt_spacing[i]:plt_spacing[i + 1], i] = np.arange(0, 1, 1. / (
                    plt_spacing[i + 1] - plt_spacing[i]))[::-1]
        elif plt_spacing[i] < spec_size:
            mel_filter[plt_spacing[i]:int(mel_to_freq(end_mel) / (rate / 2 / spec_size)), i] = \
                np.arange(0, 1, 1. / (int(mel_to_freq(end_mel) / (rate / 2 / spec_size)) - plt_spacing[i]))[::-1]
        mel_filter[plt_spacing[i], i] = 1
    # Normalize filter
    mel_filter = mel_filter / mel_filter.sum(axis=0)
    # Create and normalize inversion filter
    mel_inversion_filter = np.transpose(mel_filter) / np.transpose(mel_filter).sum(axis=0)
    mel_inversion_filter[np.isnan(mel_inversion_filter)] = 0  # for when a row has a sum of 0

    return mel_filter, mel_inversion_filter


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, 'highfreq is greater than samplerate/2'

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])

    fbank = fbank.T
    mel_inversion_filter = np.transpose(fbank) / np.transpose(fbank).sum(axis=0)
    mel_inversion_filter[np.isnan(mel_inversion_filter)] = 0  # for when a row has a sum of 0

    return fbank, mel_inversion_filter


def make_mel(spectrogram, mel_filter, shorten_factor=1):
    mel_spec = np.transpose(mel_filter).dot(np.transpose(spectrogram))
    mel_spec = scipy.ndimage.zoom(mel_spec.astype(np.float64), [1, 1. / shorten_factor]).astype(np.float64)
    mel_spec = mel_spec[:, 1:-1]  # a little hacky but seemingly needed for clipping
    return mel_spec


def mel_to_spectrogram(mel_spec, mel_inversion_filter, spec_thresh, shorten_factor):
    """
    takes in an mel spectrogram and returns a normal spectrogram for inversion
    """
    mel_spec = (mel_spec + spec_thresh)
    uncompressed_spec = np.transpose(np.transpose(mel_spec).dot(mel_inversion_filter))
    uncompressed_spec = scipy.ndimage.zoom(uncompressed_spec.astype(np.float64), [1, shorten_factor]).astype(np.float64)
    uncompressed_spec = uncompressed_spec - 4
    return uncompressed_spec


def lifter(cepstra, L=22, inverse=False):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)

        if inverse:
            lift = 1. / lift
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def mfcc(signal, sample_rate, fft_size=512, spec_thresh=4, n_mel_freq_components=256,
         shorten_factor=10, ceplifter=22):
    # Generate the mel filters
    # mel_filter, mel_inversion_filter = create_mel_filter(fft_size=fft_size,
    #                                                      n_freq_components=n_mel_freq_components,
    #                                                      start_freq=start_freq,
    #                                                      end_freq=end_freq,
    #                                                      rate=sample_rate)

    mel_filter, mel_inversion_filter = get_filterbanks(n_mel_freq_components, fft_size, sample_rate)

    step_size = fft_size // 8
    # signal = 1.0 * signal / np.max(np.abs(signal))
    signal = butter_bandpass_filter(signal, 0, sample_rate // 2, sample_rate, order=1)
    # signal = 1.0 * signal / np.max(np.abs(signal))
    wav_spectrogram, maxi = pretty_spectrogram(signal, fft_size=fft_size + 2, step_size=step_size, log=True,
                                               thresh=spec_thresh)

    mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor=shorten_factor)

    feat = mel_spec.T
    # feat = 1.0/fft_size * np.square(np.abs(feat))
    feat = feat.astype(np.float64)
    feat = dct(feat, type=2, axis=1, norm='ortho')
    # feat = feat[:, :numcep]
    feat = lifter(feat, ceplifter)
    return feat


def invmfcc(feat, sample_rate, fft_size=512, spec_thresh=4, n_mel_freq_components=256,
            shorten_factor=10, ceplifter=22):
    # Generate the mel filters
    # mel_filter, mel_inversion_filter = create_mel_filter(fft_size=fft_size,
    #                                                      n_freq_components=n_mel_freq_components,
    #                                                      start_freq=start_freq,
    #                                                      end_freq=end_freq,
    #                                                      rate=sample_rate)

    feat = lifter(feat, ceplifter, inverse=True)
    feat = idct(feat, type=2, axis=1, norm='ortho')
    feat = feat.T
    # feat = np.sqrt(fft_size * np.abs(feat))
    # full_feat = np.zeros(shape=(feat.shape[0], fft_size//2+1))
    # full_feat[:,:feat.shape[1]] = feat
    # full_feat = feat.T

    mel_filter, mel_inversion_filter = get_filterbanks(n_mel_freq_components, fft_size, sample_rate)
    step_size = fft_size // 8

    # maxi = np.max(feat)
    # mini = np.min(feat)
    #
    # print(maxi)
    # print(mini)

    # feat = lifter(feat, ceplifter)
    # feat = idct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    # mel_spec = np.exp(feat)
    # mel_spec = np.sqrt(np.abs(mel_spec - 1.0/fft_size))
    mel_inverted_spectrogram = mel_to_spectrogram(feat, mel_inversion_filter,
                                                  spec_thresh=spec_thresh,
                                                  shorten_factor=shorten_factor)
    # mel_inverted_spectrogram = mel_inverted_spectrogram.T
    # mel_inverted_spectrogram = np.exp(mel_inverted_spectrogram.T)
    # X_s = np.concatenate([mel_inverted_spectrogram, mel_inverted_spectrogram[:, ::-1]], axis=1)
    # inverted_mel_audio = iterate_invert_spectrogram(X_s, fft_size + 2, step_size, n_iter=10, verbose=False)

    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(mel_inverted_spectrogram), fft_size=fft_size + 2,
                                                   step_size=step_size, log=True, n_iter=3)

    # inverted_mel_audio = inverted_mel_audio[:-100000]
    inverted_mel_audio /= np.max(np.abs(inverted_mel_audio))
    inverted_mel_audio *= 2 ** 15
    inverted_mel_audio *= 0.9
    return inverted_mel_audio


def tf_lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        ncoeff = cepstra.get_shape()[1].value
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        lift = np.tile(lift, (cepstra.get_shape()[0].value, 1))
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def tf_dct(x, num_filters):
    ks = np.arange(num_filters)
    js = np.arange(x.get_shape()[0].value)
    js = np.tile(js, (x.get_shape()[1].value, 1))
    js = js.T
    # js = js.astype(np.float32)
    y = []
    for k in ks:
        current_transform = x * tf.cos(1.0 * k * (2 * js - 1) * np.pi / 2 / len(js))
        current_transform = tf.reduce_sum(current_transform, axis=0)
        y.append(current_transform)
    y = tf.stack(y)
    return y


def tf_make_mel(spectrogram, mel_filter, shorten_factor=1):
    mel_spec = tf.matmul(mel_filter.astype(np.float32).T, tf.transpose(spectrogram))
    # mel_spec = scipy.ndimage.zoom(mel_spec.astype(np.float64), [1, 1. / shorten_factor]).astype(np.float64)
    # mel_spec = mel_spec[:, 1:-1]  # a little hacky but seemingly needed for clipping
    return mel_spec


def tf_overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError('Window size must be even!')
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def tf_stft(x, sample_rate, framesz=0.050, hop=0.005):
    framesamp = int(framesz * sample_rate)
    hopsamp = int(hop * sample_rate)
    w = scipy.hanning(framesamp)
    x = tf.cast(x, dtype=tf.complex64)
    frame_range = x.get_shape()[0].value - framesamp
    X = [tf.fft(w * x[i:i + framesamp])
         for i in range(0, frame_range, hopsamp)]
    X = tf.stack(X)
    return X


def tf_pretty_spectrogram(d, sample_rate, log=True, thresh=5, fft_size=512, step_size=64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    # specgram = tf_stft(d, sample_rate)
    # specgram_real = tf.real(specgram)
    # specgram_real = specgram_real[:, :specgram_real.shape[1] // 2 + 1]
    # specgram_imag = tf.imag(specgram)
    # specgram_imag = specgram_imag[:, :specgram_imag.shape[1] // 2 + 1]
    # specgram_full = tf.concat_v2((specgram_real, specgram_imag), axis=1)
    specgram_full = tf_stft(d, sample_rate)
    specgram_full = specgram_full[:, :specgram_full.shape[1] // 2]
    specgram_full = tf.log1p(tf.abs(specgram_full))
    # maxi = tf.reduce_max(tf.abs(specgram_full))
    # specgram_full /= maxi  # volume normalize to max 1
    # specgram_full = tf.cast(specgram_full, dtype=tf.float32)
    # # print('Max :' + str(maxi))
    # specgram = tf.log(specgram) / tf.log(10)  # take log

    return specgram_full


def tf_mfcc(signals_tensor, sample_rate, fft_size=1102, spec_thresh=4, n_mel_freq_components=13,
            shorten_factor=10, ceplifter=22):
    signals = tf.unstack(signals_tensor, axis=0)
    mfccs = []
    for signal in signals:
        signal = tf.reshape(signal, [-1])
        # Generate the mel filters
        # mel_filter, mel_inversion_filter = create_mel_filter(fft_size=fft_size,
        #                                                      n_freq_components=n_mel_freq_components,
        #                                                      start_freq=start_freq,
        #                                                      end_freq=end_freq,
        #                                                      rate=sample_rate)

        mel_filter, mel_inversion_filter = get_filterbanks(n_mel_freq_components * 2, fft_size, sample_rate)

        step_size = fft_size // 8
        # signal = 1.0 * signal / np.max(np.abs(signal))
        # signal = butter_bandpass_filter(signal, 0, sample_rate // 2, sample_rate, order=1)
        # signal = 1.0 * signal / np.max(np.abs(signal))
        wav_spectrogram = tf_pretty_spectrogram(signal, sample_rate=sample_rate, fft_size=fft_size + 2,
                                                step_size=step_size,
                                                log=True,
                                                thresh=spec_thresh)

        # wav_spectrogram = tf.maximum(tf.log(wav_spectrogram), -200)

        # mel_spec = tf_make_mel(wav_spectrogram, mel_filter, shorten_factor=shorten_factor)
        # # mel_spec = tf.log(mel_spec + 1e-15)
        # # mel_spec = tf.transpose(mel_spec)
        # feat = tf.cast(mel_spec, tf.float64)
        # feat = tf_dct(feat, n_mel_freq_components)
        # feat = tf.transpose(feat)
        # feat = tf_lifter(feat, ceplifter)
        mfccs.append(wav_spectrogram)
    mfccs = tf.stack(mfccs)
    return mfccs
