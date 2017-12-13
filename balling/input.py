from __future__ import print_function

import logging
import math
import multiprocessing
import os
import os.path

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filenames_and_labels_in_dir(directory, extension='wav', positive_label_path_pattern='-play-'):
    r"""
    Gets all files in a directory with an optional extension filter
    :param directory: A `str`. Defines the input directory to get the files from. Will be walked.
    :param extension: An optional `str`. Allows to filter by file extension. Defaults to `wav`
    :param positive_label_path_pattern:  A `str` defines the substring contained in the path of positive examples.

    :return List[`str`] features, List[`int`] labels

    """
    all_files = []

    if extension and not extension.startswith('.'):
        extension = ''.join(['.', extension])

    for dirpath, subdirs, files in os.walk(directory):
        for file_ in files:
            if extension and file_.endswith(extension):
                all_files.append(os.path.join(dirpath, file_))
            elif extension is None:
                all_files.append(os.path.join(dirpath, file_))

    all_labels = [int(positive_label_path_pattern in filename) for filename in all_files]

    return all_files, all_labels


def parse_wav(filename):
    r"""
    Parses a wav file specified by the `filename`
    :param filename: A `str`
    :return:
    """
    wav = io_ops.read_file(filename)
    decoded_wav = contrib_audio.decode_wav(wav,
                                           desired_channels=1)
    return decoded_wav


def create_sliding_windows(decoded_wav,
                           label,
                           audio_snippet_len_secs,
                           audio_snippet_stride_secs,
                           sampling_rate_hz):
    r"""

    Creates a sliding windows over a audio tensor.

    :param decoded_wav: The decoded wav created by `tensorflow.contrib.framework.python.ops.audio_ops.decode_wav()`
    :param label: A `tf.int32` tensor with shape (). The label of the audio file
    :param audio_snippet_len_secs: An `float`. Defines how long the audio samples are.
    :param audio_snippet_stride_secs: An `float`. Defines the stride of the window over the audio.
    :param sampling_rate_hz: An `int`. Defines the sampling rate of the input audio in Hz (1/secs)
    :return:
    """

    window_size_samples = math.floor(audio_snippet_len_secs * sampling_rate_hz)
    window_stride_samples = math.floor(audio_snippet_stride_secs * sampling_rate_hz)
    windows = tf.squeeze(
        tf.extract_image_patches(decoded_wav.audio[None, ..., None],
                                 ksizes=[1, window_size_samples, 1, 1],
                                 strides=[1, window_stride_samples, 1, 1],
                                 rates=[1, 1, 1, 1],
                                 padding='VALID'))

    dataset = tf.data.Dataset.from_tensor_slices(windows)
    dataset = dataset.map(lambda window: (window, tf.identity(label)))

    return dataset


def mfcc_fingerprint_from_wav(decoded_wav,
                              window_size_secs,
                              window_stride_secs,
                              sampling_rate_hz,
                              audio_snippet_len_secs,
                              dct_coefficient_count):
    r"""
    Creates a mfcc fingerprint from a wav audio input.

    More info on mfcc: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum


    :param decoded_wav: the decoded wav file from `tensorflow.contrib.framework.python.ops`
    :param window_size_secs: the window for the audio spectrum in seconds.
    :param window_stride_secs: the stride for the audio spectrum in seconds.
    :param sampling_rate_hz: the rate at which the input was sampled in Hz.
    :param audio_snippet_len_secs: the length of the wav inputs in seconds.
    :param dct_coefficient_count: how many channels to produce per time slice.


    :return: a `Tensor` of type `tf.float32`
                with shape [time_steps, dct_coefficients, 1]
    """

    window_size_samples = sampling_rate_hz * window_size_secs
    window_stride_samples = sampling_rate_hz * window_stride_secs

    decoded_wav = tf.reshape(decoded_wav, [-1, 1])
    spectogram = contrib_audio.audio_spectrogram(
        decoded_wav,
        window_size=window_size_samples,
        stride=window_stride_samples,
        magnitude_squared=True
    )

    mfcc = contrib_audio.mfcc(
        spectogram,
        sampling_rate_hz,
        dct_coefficient_count=dct_coefficient_count
    )

    audio_snippet_len_samples = sampling_rate_hz * audio_snippet_len_secs
    overlap_between_windows_samples = window_size_samples - window_stride_samples
    spectogram_length = math.floor((audio_snippet_len_samples - overlap_between_windows_samples)
                                   / window_stride_samples)

    # reshape into two dimensions where one dimension is time and the other is the dct_coefficient_count.
    # Last dimension is channels - for now only 1.
    mfcc = tf.reshape(mfcc, (spectogram_length, dct_coefficient_count, 1))

    return mfcc


def get_input_data(input_dir,
                   batch_size,
                   epochs,
                   positive_label_path_pattern='-play-',
                   audio_snippet_len_secs=12,
                   audio_snippet_stride_secs=1,
                   sampling_rate_hz=8000,
                   spectogram_window_size_secs=0.3,
                   spectogram_stride_secs=0.1,
                   dct_coefficient_count=32,
                   shuffle_buffer_size=10000,
                   ):
    r"""

    :param input_dir: A `str`. Defines the top directory where samples should be obtained from. The path is walked.
                      Positive and negative examples are determined by pattern matching.
                      Refer to `positive_label_path_pattern`.
    :param batch_size: An `int` specifying how many samples per batch should be created.
    :param epochs: An `int` specifying the number of epochs over all samples.
    :param positive_label_path_pattern: A `str`.  Defines the pattern contained in the path to determine if an example
                                        is positive or negative. Note, that this is not used as a regex pattern
                                        matching, but merely whether this substring is contained in the path.
    :param audio_snippet_len_secs: An `float`. Defines how long the audio samples are.
    :param audio_snippet_stride_secs: An `float`. Defines the stride of the window over the audio.
    :param sampling_rate_hz: An `int`. Defines the sampling rate of the input audio in Hz (1/secs)
    :param spectogram_window_size_secs: A `float`. Defines the the windows size for creating the spectogram
    :param spectogram_stride_secs: A `float`. Defines the stride of the window for creating the spectogram
    :param dct_coefficient_count: An `int`. How many output channels the mfcc should have per time slice.
    :param shuffle_buffer_size: An `int`. Defines how many samples to queue before shuffling.
    :return: Features `Tensor` of type `tf.float32` and shape [batch_size, time_steps, dct_coefficients, 1] and
            Labels `Tensor` of type `tf.int32` and shape [batch_size]
    """

    filenames, labels = filenames_and_labels_in_dir(input_dir, positive_label_path_pattern=positive_label_path_pattern)

    filenames = tf.constant(filenames, dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda filename, label_: (parse_wav(filename),
                                                    label_),
                          num_parallel_calls=multiprocessing.cpu_count())

    dataset = dataset.flat_map(
        lambda decoded_wav, label_: create_sliding_windows(decoded_wav,
                                                           label_,
                                                           audio_snippet_len_secs=audio_snippet_len_secs,
                                                           audio_snippet_stride_secs=audio_snippet_stride_secs,
                                                           sampling_rate_hz=sampling_rate_hz))

    dataset = dataset.map(
        lambda decoded_wav, label_: (mfcc_fingerprint_from_wav(decoded_wav,
                                                               audio_snippet_len_secs=audio_snippet_len_secs,
                                                               sampling_rate_hz=sampling_rate_hz,
                                                               window_size_secs=spectogram_window_size_secs,
                                                               window_stride_secs=spectogram_stride_secs,
                                                               dct_coefficient_count=dct_coefficient_count),
                                     label_),

        num_parallel_calls=multiprocessing.cpu_count())

    dataset = dataset.cache()  # FIXME: Cache DS into memory. might not work with bigger DS.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == '__main__':

    with tf.Session() as sess:
        feature = get_input_data('data/8000hz',
                                 2,
                                 1)

        for i in range(100):
            print(sess.run([feature]))
