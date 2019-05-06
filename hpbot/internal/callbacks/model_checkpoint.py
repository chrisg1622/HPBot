import inspect
import json
from tensorflow.python import keras


class ModelCheckpointCallback(keras.callbacks.Callback):

    def __init__(self, model_directory, model_name, verbose=0):
        super(ModelCheckpointCallback, self).__init__()
        self.model_directory = model_directory
        self.model_name = model_name
        self.verbose = verbose
        self.model_filepath = f'{self.model_directory}/{self.model_name}.h5'

    @property
    def model_params(self):
        return {k: v for k, v in self.model.__dict__.items() if k in inspect.signature(self.model.__class__).parameters.keys()}

    def on_train_begin(self, logs=None):
        for name, value in self.model_params.items():
            if isinstance(value, keras.layers.Layer):
                path = f'{self.model_directory}/{name}.json'
                json.dump(value.get_config(), open(path, 'w'))
                if self.verbose > 0:
                    print(f'Saved layer params to {path}')

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.model_filepath, overwrite=True, save_format='h5')
        if self.verbose > 1:
            print(f'Saved model weights to {self.model_filepath}')
