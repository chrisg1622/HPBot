import json
import gzip
import numpy as np
from tqdm import tqdm
from language_model.data.training_sequence import TrainingSequence
from language_model.model.tasks.task import Task


class BuildTrainingSequences(Task):

    TRAINING_SEQUENCES_FILE_NAME = 'training_sequences.gz'

    def __init__(self, base_directory):
        super().__init__(base_directory=base_directory)

    def _run(self, sentence_tokens):
        np.random.shuffle(sentence_tokens)
        with gzip.open(f'{self.base_directory}/{self.TRAINING_SEQUENCES_FILE_NAME}', 'wt') as output_file:
            for tokens in tqdm(sentence_tokens, desc=self.__class__.__name__):
                if len(tokens) <= 1:
                    continue
                for target_index, target_token in enumerate(tokens, start=1):
                    training_sequence = TrainingSequence(tokens=tokens[:target_index], target_token=target_token)
                    output_file.write(json.dumps(training_sequence.to_json()))
