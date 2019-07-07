import fire
import sys
sys.path.insert(0, '../../')
from language_model.model.tasks.get_novel_sentence_tokens import GetNovelSentenceTokens
from language_model.model.tasks.train_word2vec_embeddings import TrainWord2VecEmbeddings
from language_model.model.spacy_tokenizer import SpacyTokenizer
from language_model.store.txt_novel_retriever import TxtNovelRetriever


def main(base_directory):
    get_novel_sentence_tokens = GetNovelSentenceTokens(
        base_directory=base_directory,
        novel_retriever=TxtNovelRetriever(),
        tokenizer=SpacyTokenizer()
    )
    train_word2vec_embeddings = TrainWord2VecEmbeddings(
        base_directory=base_directory,
        embedding_dimension=256,
        min_doc_frequency=1,
        iterations=20
    )
    sentence_tokens = get_novel_sentence_tokens.run()
    train_word2vec_embeddings.run(sentence_tokens=sentence_tokens)


if __name__ == '__main__':
    fire.Fire(main)
