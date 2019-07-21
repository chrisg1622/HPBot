import tensorflow as tf
from tensorflow.python import keras


class CustomLoss(keras.losses.Loss):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE, name='SparseCategoricalCrossEntropy')

    def call(self, y_true, y_pred):
        sequence_mask = tf.cast(tf.reduce_all(tf.greater(y_pred, 0.0), axis=1), dtype=tf.float32)
        sequence_cross_entropy_loss = self.loss(y_true=y_true, y_pred=y_pred) * sequence_mask
        return tf.reduce_mean(sequence_cross_entropy_loss, name='mean_sequence_cross_entropy_loss')
