import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from language_model.data.constants import START_TOKEN
from language_model.data.constants import EOS_TOKEN


class LanguageModel(keras.Model):

    def __init__(self, encoder, *args, **kwargs):
        super(LanguageModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self._vocabulary = None

    @property
    def vocabulary(self):
        if self._vocabulary is None:
            self._vocabulary = {term: ind for ind, term in enumerate(self.encoder.vocabulary)}
        return self._vocabulary

    def call(self, sentence_tokens, *args, **kwargs):
        return self.encoder(inputs=sentence_tokens)

    def sample_next_word(self, tokens, top_k=3, greedy=False):
        sentence_token_softmaxed_logits = self.call(sentence_tokens=np.array([tokens]), training=False)
        top_logits, top_indices = tf.math.top_k(input=sentence_token_softmaxed_logits[0, :], k=top_k)
        if greedy:
            return self.encoder.vocabulary[top_indices[0].numpy()]
        softmaxed_top_values = tf.nn.softmax(logits=top_logits)
        top_words = [self.encoder.vocabulary[k] for k in top_indices.numpy()]
        return np.random.choice(top_words, size=1, p=softmaxed_top_values.numpy())[0]

    def sample_sequence(self, words=None, greedy=False, max_sample_length=50, top_k=3):
        sample = words or [START_TOKEN]
        last_word = sample[-1]
        while last_word != EOS_TOKEN:
            last_word = self.sample_next_word(tokens=sample, greedy=greedy, top_k=top_k)
            sample.append(last_word)
            if len(sample) == max_sample_length:
                break
        return self.format_sequence(sample)

    @staticmethod
    def format_sequence(sequence):
        if START_TOKEN in sequence:
            sequence.remove(START_TOKEN)
        formatted_sequence = sequence[0]
        for token in sequence[1:]:
            last_char = formatted_sequence[-1]
            if not(token[0].isalpha()):
                if last_char.isalpha():
                    formatted_sequence += token
                    continue
            formatted_sequence += f' {token}'
        return formatted_sequence

