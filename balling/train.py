import tensorflow as tf
from balling.models import cnn_model
from balling import input

classifier = tf.estimator.Estimator(model_fn=cnn_model,
                                    model_dir='/tmp/pingis_train')
experiment = tf.contrib.learn.Experiment(classifier,
                                         eval_steps=100,
                                         train_input_fn=lambda: input.get_input_data('data/8000hz',
                                                                                     batch_size=256,
                                                                                     epochs=1000),
                                         eval_input_fn=lambda: input.get_input_data('data/8000hz',
                                                                                     batch_size=256,
                                                                                     epochs=1000))

if __name__ == '__main__':
    experiment.train_and_evaluate()
