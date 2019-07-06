import tensorflow as tf
from tensorflow.python import keras


class SparseCategoricalAccuracy(keras.metrics.Metric):

    def __init__(self, name='sparse_categorical_accuracy', **kwargs):
        super(SparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sentence_token_mask = tf.cast(tf.reduce_all(tf.greater(y_pred, 0.0), axis=2), dtype=tf.float32)
        sentence_token_predictions = tf.cast(tf.argmax(y_pred, axis=2, output_type=tf.int32), dtype=tf.float32)
        sentence_token_match = tf.cast(tf.equal(sentence_token_predictions, y_true), dtype=tf.float32) * sentence_token_mask
        count = tf.reduce_sum(sentence_token_match)
        total = tf.reduce_sum(sentence_token_mask)
        self.count.assign_add(count)
        self.total.assign_add(total)

    def result(self):
        return self.count / tf.maximum(self.total, 1.0)
