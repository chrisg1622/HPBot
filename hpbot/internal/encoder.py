import tensorflow as tf
from tensorflow.python import keras


class Encoder(keras.layers.Layer):

    def __init__(self, vocabulary, embedding_dimension, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.vocabulary = vocabulary
        self.embedding_dimension = embedding_dimension
        self.vocab_table = None
        self.embeddings = None
        self.lstm = None
        self.output_layer = None

    def build(self, input_shape):
        table_initializer = tf.lookup.KeyValueTensorInitializer(keys=self.vocabulary, values=list(range(len(self.vocabulary))), key_dtype=tf.string, value_dtype=tf.int32, name='vocab_table_init')
        self.vocab_table = tf.lookup.StaticHashTable(initializer=table_initializer, default_value=0, name='vocab_table')
        self.embeddings = keras.layers.Embedding(input_dim=len(self.vocabulary), output_dim=self.embedding_dimension, name='Embeddings')
        self.lstm = keras.layers.LSTM(units=self.embedding_dimension, name='stacked_lstm')
        self.output_layer = keras.layers.Dense(units=len(self.vocabulary), activation='sigmoid')

    def call(self, inputs, *args, **kwargs):
        token_sequences = inputs
        token_id_sequences = self.vocab_table.lookup(token_sequences)
        token_embeddings = self.embeddings(inputs=token_id_sequences)
        lstm_output = self.lstm(inputs=token_embeddings)
        return self.output_layer(inputs=lstm_output)
