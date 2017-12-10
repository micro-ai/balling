import tensorflow as tf
from balling import input


def fc_model(input_dir, learning_rate, batch_size, epochs):

    features, label = input.get_input_data(input_dir, batch_size, epochs)


tf.estimator.Estimator()