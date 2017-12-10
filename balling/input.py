from __future__ import print_function

import logging
import os
import os.path
import subprocess

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_with_logging(cmd):
    """
    Run cmd and wait for it to finish. While cmd is running, we read it's
    output and print it to a logger.

    :param cmd: command to run. Should be a list of command tokens, e.g. ['ls', '-l']
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        line = process.stdout.readline()
        if not line:
            break
        logger.info(line.rstrip('\n'))

    exit_code = process.wait()

    if exit_code:
        raise subprocess.CalledProcessError(exit_code, cmd)

    return exit_code


def split_audio_files(path, outpath, extension=None, segment_duration=12):
    """
    Splits audio files into time segments of seconds. The output files are named after the input file, but save
    in a specified directory

    :param path: path to input files. Can be a directory or a single file
    :param outpath: directory where output file will be written to
    :param segment_duration: duration of each segment in seconds
    :return:
    """

    if extension and not extension.endswith('.'):
        extension = ''.join(['.', extension])

    if os.path.isfile(path):
        files = [path]
    else:
        files = filenames_in_dir(path, extension)

    part_suffix = '%06d'

    for filepath in files:
        if extension:
            outfile = os.path.join(
                outpath,
                ''.join([os.path.basename(filepath[0:-len(extension)]), part_suffix, extension])
            )

        else:
            outfile = os.path.join(
                outpath,
                ''.join([os.path.basename(filepath), part_suffix])
            )

        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

        cmd = [
            'ffmpeg',
            '-i',
            filepath,
            '-f',
            'segment',
            '-segment_time',
            str(segment_duration),
            '-c',
            'copy',
            outfile
        ]

        run_with_logging(cmd)


def filenames_in_dir(directory, extension=None):
    r"""
    Gets all files in a directory with an optional extension filter
    :param directory: input directory to get the files from
    :param extension: `optional` filter by extension
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
    return all_files


def mfcc_fingerprint_from_wav(filename, window_size_secs, stride_secs, sampling_rate_hz, audio_length_secs):
    r"""
    Creates a mfcc fingerprint from a wav audio input.

    More info on mfcc: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum


    :param filename: filename of the wav file
    :param window_size_secs: the window for the audio spectrum in seconds.
    :param stride_secs: the stride for the audio spectrum in seconds.
    :param sampling_rate_hz: the rate at which the input was sampled in Hz.
    :param audio_length_secs: the length of the wav inputs in seconds.

    :return: a single mfcc representation of the input audio file


    """

    # TODO: There should be a more efficient way of doing this.

    desired_samples = audio_length_secs * sampling_rate_hz
    window_size_samples = sampling_rate_hz * window_size_secs
    stride_samples = sampling_rate_hz * stride_secs

    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader,
                                               desired_channels=1,
                                               desired_samples=desired_samples)
        spectogram = contrib_audio.audio_spectrogram(
            wav_decoder.audio,
            window_size=window_size_samples,
            stride=stride_samples,
            magnitude_squared=True
        )

        mfcc = contrib_audio.mfcc(
            spectogram,
            sampling_rate_hz
        )

        return sess.run(
            mfcc,
            feed_dict={wav_filename_placeholder: filename}
        )


def audio_tensors_generator(directory,
                            window_size_secs=0.03,
                            stride_secs=0.01,
                            sampling_rate_hz=8000,
                            audio_length_secs=12,
                            positive_label_dir_name='-play-',
                            ):
    r"""
    creates a generator from a directory with PCM coded tensors of all wav files
    in the directory

    :param directory the directory: from where to load the files
    :param positive_label_dir_name: specifies a string that needs to be present in the path when it is a positive label
    """
    filenames = filenames_in_dir(directory, extension='wav')

    for filename in filenames:
        label = int(positive_label_dir_name in filename)
        yield (mfcc_fingerprint_from_wav(filename,
                                         window_size_secs,
                                         stride_secs,
                                         sampling_rate_hz,
                                         audio_length_secs),
               label)


def get_input_data(input_dir, batch_size, repeat, buffer_size=1000):
    r"""
    :param input_dir:
    :param batch_size:
    :param repeat:
    :param buffer_size:
    :return:
    """
    dataset = tf.data.Dataset.from_generator(lambda: audio_tensors_generator(input_dir),
                                             output_types=(tf.float32, tf.int64),
                                             output_shapes=(tf.TensorShape([1, None, 13]), tf.TensorShape([]))
                                             )

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == '__main__':

    # for i in audio_tensors_generator('data/splits'):
    #     print(i)

    with tf.Session() as sess:
        feature, label = get_input_data('data/splits',
                                        3,
                                        200)
        for i in range(100):
            print(sess.run([feature, label]))
