import json
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from language_model.model.encoder import Encoder
from language_model.model.language_model import LanguageModel
from language_model.model.losses.custom_loss import CustomLoss
from language_model.model.callbacks.sequence_sampling import SequenceSamplingCallback
from language_model.model.callbacks.model_checkpoint import ModelCheckpointCallback
from language_model.model.tasks.train_word2vec_embeddings import TrainWord2VecEmbeddings
from language_model.model.tasks.build_training_sequences import BuildTrainingSequences
from language_model.model.tasks.task import Task
from language_model.store.training_sequence_retriever import TrainingSequenceRetriever


class TrainLanguageModel(Task):

    def __init__(self, base_directory, model_name, hidden_size, restore_model):
        super().__init__(base_directory=base_directory)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.restore_model = restore_model

    def _run(self, batch_size, learning_rate, epochs, batches_per_epoch):
        vocabulary = json.load(open(f'{self.base_directory}/{TrainWord2VecEmbeddings.VOCAB_FILE_NAME}'))
        training_sequence_retriever = TrainingSequenceRetriever(
            vocabulary=vocabulary,
            training_sequence_file_path=f'{self.base_directory}/{BuildTrainingSequences.TRAINING_SEQUENCES_FILE_NAME}'
        )
        hp_bot = LanguageModel(
            encoder=Encoder(
                vocabulary=vocabulary,
                hidden_size=self.hidden_size,
                pretrained_embeddings=np.load(f'{self.base_directory}/{TrainWord2VecEmbeddings.VECTORS_FILE_NAME}')
            )
        )
        hp_bot.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=CustomLoss(), metrics=[])
        _ = hp_bot(np.array([['', '', '']]))
        if self.restore_model:
            hp_bot.load_weights(filepath=f'{self.base_directory}/{self.model_name}.h5')
        hp_bot.fit(
            x=training_sequence_retriever.get_generator(),
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=batches_per_epoch,
            workers=0,
            callbacks=[
                ModelCheckpointCallback(
                    model_directory=self.base_directory,
                    model_name=self.model_name,
                    as_pickle=True,
                    verbose=1
                ),
                keras.callbacks.TensorBoard(
                    log_dir=f'{self.base_directory}/tensorboard',
                    write_graph=True,
                    update_freq='batch',
                    profile_batch=False
                ),
                SequenceSamplingCallback(
                    max_sequence_length=30,
                    greedy_sequence=False,
                    name='non-greedy sample',
                    logger=self.logger
                ),
                SequenceSamplingCallback(
                    max_sequence_length=30,
                    greedy_sequence=True,
                    name='greedy sample',
                    logger=self.logger
                )
            ]
        )
