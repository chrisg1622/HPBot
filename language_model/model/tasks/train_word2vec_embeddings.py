import json
import numpy as np
from gensim.models import Word2Vec
from language_model.model.tasks.task import Task


class TrainWord2VecEmbeddings(Task):

    MODEL_FILE_NAME = 'word2vec_model'
    VECTORS_FILE_NAME = 'word2vec_vectors.npy'
    VOCAB_FILE_NAME = 'word2vec_vocab.json'

    def __init__(self, base_directory, embedding_dimension, min_doc_frequency, iterations, **kwargs):
        super().__init__(base_directory=base_directory, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.min_doc_frequency = min_doc_frequency
        self.iterations = iterations

    def run(self, sentence_tokens):
        w2v = Word2Vec(size=self.embedding_dimension, min_count=self.min_doc_frequency, iter=self.iterations)
        w2v.build_vocab(sentences=sentence_tokens)
        w2v.train(sentence_tokens, compute_loss=True, total_examples=w2v.corpus_count, epochs=w2v.iter)
        w2v.save(f'{self.base_directory}/{self.MODEL_FILE_NAME}')
        np.save(f'{self.base_directory}/{self.VECTORS_FILE_NAME}', w2v.wv.vectors)
        json.dump(w2v.wv.index2word, open(f'{self.base_directory}/{self.VOCAB_FILE_NAME}', 'w'))
        for word in np.random.choice(a=w2v.wv.index2word, size=5):
            self.log(message=f'\nWord: {word}\nMost similar: {w2v.similar_by_word(word=word)}')
