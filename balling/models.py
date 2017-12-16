import tensorflow as tf

def cnn_model(features, labels, params=None, config=None):
    net = tf.layers.conv2d(features,
                           kernel_size=(3, 3),
                           filters=64,
                           activation=tf.nn.selu,
                           name='conv_1')
    net = tf.layers.conv2d(net,
                           kernel_size=(3, 3),
                           filters=32,
                           activation=tf.nn.selu,
                           name='conv_2')
    net = tf.layers.conv2d(net,
                           kernel_size=(3, 3),
                           filters=16,
                           activation=tf.nn.selu,
                           name='conv_3')
    net = tf.layers.conv2d(net,
                           kernel_size=(3, 3),
                           filters=2,
                           activation=tf.nn.selu,
                           name='conv_4')
    logits = tf.reduce_mean(net, axis=(1, 2))

    predictions = tf.argmax(tf.nn.softmax(logits), axis=1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions),
    }

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)

