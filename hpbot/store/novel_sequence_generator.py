import numpy as np
from tqdm.auto import tqdm


class NovelSequenceGenerator:

    OOV_TOKEN = '<OOV>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def pad_tokens(self, tokens):
        return [self.START_TOKEN] + tokens + [self.END_TOKEN]

    def get_token_ngrams(self, tokens, n, pad_tokens=False):
        input_list = self.pad_tokens(tokens=tokens) if pad_tokens else tokens
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_chapter_ngrams(self, chapter, n):
        tokens = self.get_chapter_tokens(chapter=chapter)
        return self.get_token_ngrams(tokens=tokens, n=n)

    def get_chapter_tokens(self, chapter, with_title=True):
        title_tokens = self.tokenizer.tokenize(chapter.title) if with_title else []
        return title_tokens + self.tokenizer.tokenize(chapter.text)

    def get_chapter_sentence_tokens(self, chapter, with_title=True, pad_tokens=True):
        title_sentence_tokens = self.tokenizer.get_sentence_tokens(chapter.title) if with_title else []
        if with_title and pad_tokens:
            title_sentence_tokens = [self.pad_tokens(tokens=tokens) for tokens in title_sentence_tokens]
        chapter_tokens = self.tokenizer.get_sentence_tokens(chapter.text)
        if pad_tokens:
            chapter_tokens = [self.pad_tokens(tokens=tokens) for tokens in chapter_tokens]
        return title_sentence_tokens + chapter_tokens

    def get_novel_tokens(self, novel):
        return sum([self.get_chapter_tokens(chapter=chapter) for chapter in tqdm(novel.chapters, desc=novel.title)], [])

    def get_novel_sentence_tokens(self, novel, pad_tokens=True):
        return sum([self.get_chapter_sentence_tokens(chapter, pad_tokens=pad_tokens) for chapter in tqdm(novel.chapters, desc=novel.title)], [])

    def get_novel_ngrams(self, novel, n):
        return [
            ngram
            for chapter in tqdm(novel.chapters)
            for ngram in self.get_chapter_ngrams(chapter=chapter, n=n)
        ]

    def get_novel_vocabulary(self, novels):
        defaultTokens = [self.OOV_TOKEN, self.START_TOKEN, self.END_TOKEN]
        novelTokens = sorted(list(set().union(*[set(self.get_novel_tokens(novel=novel)) for novel in novels])))
        return defaultTokens + novelTokens

    def get_novel_training_ngrams(self, novels, n):
        sequences = sum([self.get_novel_ngrams(novel, n=n+1) for novel in novels], [])
        return np.array([ngram[:n] for ngram in sequences]), np.array([ngram[n] for ngram in sequences])

    def get_novels_sentence_tokens(self, novels, pad_tokens=True):
        return sum([self.get_novel_sentence_tokens(novel=novel, pad_tokens=pad_tokens) for novel in novels], [])

    def _pad_sentence(self, sentence, max_size, padding=''):
        return sentence + [padding]*(max_size - len(sentence))

    def get_training_sequences(self, sentence_tokens):
        max_sequence_length = max([len(sentence) for sentence in sentence_tokens]) - 1
        input_sequences = np.array([self._pad_sentence(sentence=tokens[:-1], max_size=max_sequence_length) for tokens in sentence_tokens])
        output_sequences = np.array([self._pad_sentence(sentence=tokens[1:], max_size=max_sequence_length) for tokens in sentence_tokens])
        return input_sequences, output_sequences

    def get_training_sequence_generator(self, training_sequences, batch_size):
        pass
