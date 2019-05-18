import tensorflow as tf
from tensorflow.python import keras


class CustomLoss(keras.losses.Loss):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE, name='SparseCategoricalCrossEntropy')

    def call(self, y_true, y_pred):
        sentence_token_mask = tf.cast(tf.reduce_all(tf.greater(y_pred, 0.0), axis=2), dtype=tf.float32)
        sentence_lengths = tf.reduce_sum(sentence_token_mask, axis=1)
        sentence_token_cross_entropy_loss = self.loss(y_true=y_true, y_pred=y_pred, sample_weight=sentence_token_mask)
        sentence_loss = tf.reduce_sum(sentence_token_cross_entropy_loss, axis=1) / tf.maximum(sentence_lengths, 1.0)
        sentence_token_predictions = tf.argmax(y_pred, axis=2)
        common_prediction_sentence_loss = tf.reduce_sum(tf.cast(tf.equal(sentence_token_predictions[:, :-1], sentence_token_predictions[:, 1:]), dtype=tf.float32), axis=1) * sentence_loss
        return tf.reduce_mean(sentence_loss + common_prediction_sentence_loss, name='mean_sentence_cross_entropy_loss')
