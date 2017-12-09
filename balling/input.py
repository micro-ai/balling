import tensorflow as tf


def input_wav_files(input_dir):
    r"""
    :param input_dir: string of where the input data is stored.
    """
    dataset = tf.data.Dataset()