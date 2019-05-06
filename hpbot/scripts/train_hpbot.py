import fire
import json
import tensorflow as tf
from tensorflow.python import keras
from hpbot.internal.spacy_tokenizer import SpacyTokenizer
from hpbot.internal.encoder import Encoder
from hpbot.internal.hpbot import HPBot
from hpbot.internal.callbacks.sequence_sampling import SequenceSamplingCallback
from hpbot.internal.callbacks.model_checkpoint import ModelCheckpointCallback
from hpbot.store.txt_novel_retriever import TxtNovelRetriever
from hpbot.store.novel_sequence_generator import NovelSequenceGenerator


txt_novel_retriever = TxtNovelRetriever()
spacy_tokenizer = SpacyTokenizer()
sequence_generator = NovelSequenceGenerator(tokenizer=spacy_tokenizer)
repository_path = '/Users/cgeorge/Git/HPBot'
model_name = 'HPBot'
optimizer = tf.optimizers.Adam(learning_rate=0.007)
loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = [keras.metrics.SparseCategoricalAccuracy()]


def main(sequence_size=4, batch_size=64, epochs=5, novel_number=1, restore_model=True):
    save_dir = f'{repository_path}/models/novel{novel_number}'
    tensorboard_dir = f'{repository_path}/tensorboard/novel{novel_number}'
    novel, lines = txt_novel_retriever.retrieve_novel(f'{repository_path}/Data/HP{novel_number}.txt')
    if restore_model:
        encoder = Encoder.from_config(config=json.load(open(f'{save_dir}/Encoder.json', 'r')))
    else:
        vocabulary = sequence_generator.get_novel_vocabulary(novel=novel)
        encoder = Encoder(vocabulary=vocabulary, embedding_dimension=300)
    hp_bot = HPBot(encoder=encoder, sequence_size=sequence_size)

    token_sequences, target_tokens = sequence_generator.get_novel_training_ngrams(novel=novel, n=sequence_size)
    target_indices = hp_bot.get_target_indices(tokens=target_tokens)
    i=0
    for a,b,c in zip(token_sequences, target_tokens, target_indices):
        print(a,b,c)
        i+=1
        if i == 1000:
            return
    hp_bot.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    _ = hp_bot(token_sequences[:1])
    if restore_model:
        hp_bot.load_weights(filepath=f'{save_dir}/{model_name}.h5')
    callbacks = [
        ModelCheckpointCallback(model_directory=save_dir, model_name=model_name, verbose=1),
        keras.callbacks.TensorBoard(log_dir=tensorboard_dir, profile_batch=False),
        SequenceSamplingCallback(sequence_length=30, greedy_sequence=False, name='non-greedy sample'),
        SequenceSamplingCallback(sequence_length=30, greedy_sequence=True, name='greedy sample')
    ]
    hp_bot.fit(
        x=token_sequences,
        y=target_indices,
        batch_size=batch_size,
        epochs=epochs,
        workers=4,
        validation_split=0.1,
        callbacks=callbacks
    )


if __name__ == '__main__':
    fire.Fire(main)
