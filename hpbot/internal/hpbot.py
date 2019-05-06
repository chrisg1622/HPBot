import numpy as np
from tensorflow.python import keras
from hpbot.store.novel_sequence_generator import NovelSequenceGenerator


class HPBot(keras.Model):

    START_TOKEN = NovelSequenceGenerator.START_TOKEN

    def __init__(self, encoder, sequence_size, *args, **kwargs):
        super(HPBot, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.sequence_size = sequence_size
        self._vocabulary = None

    @property
    def vocabulary(self):
        if self._vocabulary is None:
            self._vocabulary = {term: ind for ind, term in enumerate(self.encoder.vocabulary)}
        return self._vocabulary

    def call(self, inputs, *args, **kwargs):
        return self.encoder(inputs)

    def get_target_indices(self, tokens):
        return np.array([self.vocabulary.get(tok, 0) for tok in tokens])

    def sample_next_word(self, words, greedy=False):
        input_seq = np.array([[self.START_TOKEN] * (self.sequence_size - len(words)) + words])
        softmaxed_logits = self.call(inputs=input_seq)
        if greedy:
            return self.encoder.vocabulary[np.argmax(softmaxed_logits[0])]
        return np.random.choice(self.encoder.vocabulary, size=1, p=softmaxed_logits.numpy()[0])[0]

    def sample_sequence(self, words=None, size=20, greedy=False):
        sample = words or [self.START_TOKEN]
        for i in range(size):
            words = sample if len(sample) < self.sequence_size else sample[-self.sequence_size:]
            sample.append(self.sample_next_word(words=words, greedy=greedy))
        return self.format_sequence(sample)

    def format_sequence(self, sequence):
        if self.START_TOKEN in sequence:
            sequence.remove(self.START_TOKEN)
        formatted_sequence = sequence[0]
        for token in sequence[1:]:
            last_char = formatted_sequence[-1]
            if not(token[0].isalpha()):
                if last_char.isalpha():
                    formatted_sequence += token
                    continue
            formatted_sequence += f' {token}'
        return formatted_sequence

