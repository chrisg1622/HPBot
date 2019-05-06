import fire
import tensorflow as tf
from hpbot.internal.spacy_tokenizer import SpacyTokenizer
from hpbot.internal.encoder import Encoder
from hpbot.internal.hpbot import HPBot
from hpbot.store.txt_novel_retriever import TxtNovelRetriever
from hpbot.store.novel_sequence_generator import NovelSequenceGenerator


txt_novel_retriever = TxtNovelRetriever()
spacy_tokenizer = SpacyTokenizer()
sequence_generator = NovelSequenceGenerator(tokenizer=spacy_tokenizer)
base_save_dir = '/Users/cgeorge/Git/models/HPBot'
iterations = 200000


def main(sequence_size=4, batch_size=32, save_dir=None):
    save_dir = save_dir or base_save_dir
    novel, lines = txt_novel_retriever.retrieve_novel('/Users/cgeorge/Git/Data/HPBot/HP1.txt')
    vocabulary = sequence_generator.get_novel_vocabulary(novel=novel)
    encoder = Encoder(vocabulary=vocabulary, embedding_dimension=200)
    hp_bot = HPBot(encoder=encoder, sequence_size=sequence_size)
    token_sequences, target_tokens = sequence_generator.get_novel_training_ngrams(novel=novel, n=sequence_size)
    target_indices = hp_bot.get_target_indices(tokens=target_tokens)

    optimizer = tf.optimizers.Adam(learning_rate=0.007)
    loss_function = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    hp_bot.compile(optimizer=optimizer, loss=loss_function)
    for epoch in range(len(target_indices) // batch_size):
        sample_x = token_sequences[batch_size * epoch: batch_size * (epoch + 1)]
        sample_t = target_indices[batch_size * epoch: batch_size * (epoch + 1)]
        with tf.GradientTape() as tape:
            sample_y = hp_bot(sample_x)
            loss = loss_function(sample_t, sample_y)
            grads = tape.gradient(loss, hp_bot.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, hp_bot.trainable_variables)))
        if epoch % 50 == 0:
            print('Epoch {} - Loss:{}'.format(epoch, loss))
            print(' '.join(hp_bot.sample_sequence()))
            hp_bot.save_weights(filepath=f'{save_dir}/{epoch}/HPBot')


if __name__ == '__main__':
    fire.Fire(main)
