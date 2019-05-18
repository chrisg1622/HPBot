import tensorflow as tf
from tensorflow.python import keras


class Encoder(keras.layers.Layer):

    def __init__(self, vocabulary, hidden_size, pretrained_embeddings=None, regularizer=1.0, dropout=0.4, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        self.pretrained_embeddings = pretrained_embeddings
        self.regularizer = regularizer
        self.dropout = dropout
        self.vocab_table = None
        self.embeddings = None
        self.lstm = None
        self.output_layer = None

    def build(self, input_shape):
        table_initializer = tf.lookup.KeyValueTensorInitializer(keys=self.vocabulary, values=list(range(len(self.vocabulary))), key_dtype=tf.string, value_dtype=tf.int32, name='vocab_table_init')
        self.vocab_table = tf.lookup.StaticHashTable(initializer=table_initializer, default_value=0, name='vocab_table')
        if self.pretrained_embeddings is None:
            self.embeddings = keras.layers.Embedding(
                input_dim=len(self.vocabulary) + 1,
                output_dim=self.hidden_size,
                mask_zero=True
            )
        self.lstm = keras.layers.LSTM(units=self.hidden_size, return_sequences=True, dropout=self.dropout, name='stacked_lstm')
        self.output_layer = keras.layers.Dense(
            units=len(self.vocabulary),
            use_bias=True,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(l=self.regularizer)
        )

    def call(self, inputs, *args, **kwargs):
        sentence_tokens = inputs
        sentence_token_mask = tf.not_equal(sentence_tokens, '', name='sentenceTokenMask')
        sentence_token_ids = self.vocab_table.lookup(sentence_tokens)
        if self.pretrained_embeddings is not None:
            sentence_token_embeddings = tf.gather(self.pretrained_embeddings, sentence_token_ids)
        else:
            sentence_token_embeddings = self.embeddings((sentence_token_ids + 1) * tf.cast(sentence_token_mask, dtype=tf.int32))
        sentence_token_lstm_output = self.lstm(inputs=sentence_token_embeddings, mask=sentence_token_mask, training=kwargs.get('training'))
        return self.output_layer(inputs=sentence_token_lstm_output) * tf.expand_dims(tf.cast(sentence_token_mask, dtype=tf.float32), axis=2)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config['vocabulary'] = self.vocabulary
        config['hidden_size'] = self.hidden_size
        config['pretrained_embeddings'] = self.pretrained_embeddings
        config['regularizer'] = self.regularizer
        config['dropout'] = self.dropout
        return config
