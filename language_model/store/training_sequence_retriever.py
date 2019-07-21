import json
import gzip
import numpy as np
from language_model.data.training_sequence import TrainingSequence


class TrainingSequenceRetriever:

    def __init__(self, vocabulary, training_sequence_file_path):
        self.training_sequence_file_path = training_sequence_file_path
        self.vocabulary = vocabulary
        self.wordIndices = {term: ind for ind, term in enumerate(self.vocabulary)}

    def get_generator(self):
        with gzip.open(filename=self.training_sequence_file_path, mode='rt') as inputFile:
            for line in inputFile:
                trainingSequence = TrainingSequence.from_json(json_object=json.loads(line.strip('\n')))
                yield np.array([trainingSequence.tokens]), np.array([self.wordIndices[trainingSequence.target_token]])
