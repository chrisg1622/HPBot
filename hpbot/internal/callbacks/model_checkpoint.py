from tensorflow.python.keras.callbacks import Callback


class SequenceSamplingCallback(Callback):

    def __init__(self, sequence_length, samples=1, greedy_sequence=False, name=None):
        super(SequenceSamplingCallback, self).__init__()
        self.sequence_length = sequence_length
        self.samples = samples
        self.greedy_sequence = greedy_sequence
        self.name = name or 'sample'

    def print_sample(self):
        for i in range(self.samples):
            sample = self.model.sample_sequence(size=self.sequence_length, greedy=self.greedy_sequence)
            print(f'{self.name} {i}: {sample}')

    def on_train_begin(self, logs=None):
        self.print_sample()

    def on_epoch_end(self, epoch, logs=None):
        self.print_sample()

