import fire
import json
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from hpbot.model.spacy_tokenizer import SpacyTokenizer
from hpbot.model.encoder import Encoder
from hpbot.model.hpbot import HPBot
from hpbot.model.custom_loss import CustomLoss
from hpbot.model.callbacks.sequence_sampling import SequenceSamplingCallback
from hpbot.model.callbacks.model_checkpoint import ModelCheckpointCallback
from hpbot.store.txt_novel_retriever import TxtNovelRetriever
from hpbot.store.novel_sequence_generator import NovelSequenceGenerator


txt_novel_retriever = TxtNovelRetriever()
spacy_tokenizer = SpacyTokenizer()
sequence_generator = NovelSequenceGenerator(tokenizer=spacy_tokenizer)
repository_path = '/Users/cgeorge/Git/HPBot'
model_name = 'HPBot'
optimizer = tf.optimizers.Adam(learning_rate=0.007)
loss_function = CustomLoss()
metrics = []

save_dir = f'{repository_path}/models/global'
tensorboard_dir = f'{repository_path}/tensorboard/global'
vocabulary = json.load(open(f'{repository_path}/models/word2vec_vocab.json'))
embeddings = np.load(f'{repository_path}/models/word2vec_vectors.npy')


def main(batch_size=32, epochs=1, batches_per_epoch=200, restore_model=False):
    sentence_tokens = json.load(open(f'{repository_path}/data/sentence_tokens.json', 'r'))
    encoder = Encoder(vocabulary=vocabulary, hidden_size=256, pretrained_embeddings=embeddings)
    hp_bot = HPBot(encoder=encoder)
    print(f'{len(sentence_tokens)} sentences')
    input_generator = sequence_generator.get_training_sequence_generator(sentence_tokens=sentence_tokens, target_wrapper_fn=hp_bot.get_sentence_token_indices)
    hp_bot.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    _ = hp_bot(np.array([['', '', '']]))
    if restore_model:
        hp_bot.load_weights(filepath=f'{save_dir}/{model_name}.h5')
    callbacks = [
        ModelCheckpointCallback(model_directory=save_dir, model_name=model_name, as_pickle=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=tensorboard_dir, profile_batch=False),
        SequenceSamplingCallback(max_sequence_length=30, greedy_sequence=False, name='non-greedy sample'),
        SequenceSamplingCallback(max_sequence_length=30, greedy_sequence=True, name='greedy sample')
    ]
    hp_bot.fit(
        x=input_generator,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=batches_per_epoch,
        workers=1,
        callbacks=callbacks
    )


if __name__ == '__main__':
    fire.Fire(main)
