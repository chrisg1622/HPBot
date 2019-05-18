from tensorflow.python.keras.callbacks import Callback


class SequenceSamplingCallback(Callback):

    def __init__(self, max_sequence_length, samples=1, greedy_sequence=False, name=None):
        super(SequenceSamplingCallback, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.samples = samples
        self.greedy_sequence = greedy_sequence
        self.name = name or 'sample'

    def print_sample(self):
        for i in range(self.samples):
            sample = self.model.sample_sequence(words=None, greedy=self.greedy_sequence, max_sample_length=self.max_sequence_length)
            print(f'\n{self.name} {i}: {sample}\n')

    def on_train_begin(self, logs=None):
        self.print_sample()

    def on_epoch_end(self, epoch, logs=None):
        self.print_sample()

