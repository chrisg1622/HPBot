import json
import gzip
import numpy as np
from itertools import islice
from language_model.data.training_sequence import TrainingSequence


class TrainingSequenceRetriever:

    def __init__(self, vocabulary, training_sequence_file_path):
        self.training_sequence_file_path = training_sequence_file_path
        self.vocabulary = vocabulary
        self.wordIndices = {term: ind for ind, term in enumerate(self.vocabulary)}

    def get_generator(self):
        with gzip.open(filename=self.training_sequence_file_path, mode='rt') as inputFile:
            for line in inputFile:
                training_sequence = TrainingSequence.from_json(json_object=json.loads(line.strip('\n')))
                yield training_sequence

    def pad_tokens_batch(self, tokens_batch):
        max_length = max([len(tokens) for tokens in tokens_batch])
        return np.array([tokens + [''] * (max_length - len(tokens)) for tokens in tokens_batch])

    def format_training_sequences(self, training_sequences):
        tokens_batch = np.array([training_sequence.tokens for training_sequence in training_sequences])
        target_token_indices_batch = np.array([self.wordIndices[training_sequence.target_token] for training_sequence in training_sequences])
        return self.pad_tokens_batch(tokens_batch=tokens_batch), target_token_indices_batch

    def get_batched_generator(self, batch_size):
        generator = self.get_generator()
        while True:
            training_sequences = list(islice(generator, batch_size))
            if len(training_sequences) == 0:
                break
            yield self.format_training_sequences(training_sequences=training_sequences)
