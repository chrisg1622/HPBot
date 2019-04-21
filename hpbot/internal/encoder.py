import tensorflow.python as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, vocabulary, embedding_dimension, stack_depth, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.vocabulary = vocabulary
        self.embedding_dimension = embedding_dimension
        self.stack_depth = stack_depth
        self.vocab_table = None
        self.stacked_lstm = None

    def __build__(self, input_shape):
        table_initializer = tf.lookup.KeyValueTensorInitializer(keys=self.vocabulary, values=list(range(len(self.vocabulary))), key_dtype=tf.string, value_dtype=tf.int32, name='vocab_table_init')
        self.vocab_table = tf.lookup.StaticHashTable(initializer=table_initializer, default_value=0, name='vocab_table')
        self.embeddings = tf.keras.layers.Embedding(input_dim=len(self.vocabulary), output_dim=self.embedding_dimension, name='Embeddings')
        self.stacked_lstm = tf.keras.layers.StackedRNNCells(cells=[tf.python.keras.layers.LSTMCell(units=self.embedding_dimension)]*self.stack_depth, name='stacked_lstm')
        self.output_layer = tf.keras.layers.Dense(units=len(self.vocabulary), activation='sigmoid')

    def __call__(self, inputs, *args, **kwargs):
        token_sequences = inputs
        token_id_sequences = self.vocab_table.lookup(token_sequences)
        token_embeddings = self.embeddings(inputs=token_id_sequences)
        lstm_output = self.stacked_lstm(inputs=token_embeddings)
        return self.output_layer(inputs=lstm_output)

