import numpy as np
from tqdm.auto import tqdm


class NovelSequenceGenerator:

    OOV_TOKEN = '<OOV>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_chapter_ngrams(self, chapter, n):
        tokens = self.get_chapter_tokens(chapter=chapter)
        input_list = [self.START_TOKEN] * (n - 1) + tokens + [self.END_TOKEN]
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_chapter_tokens(self, chapter, with_title=True):
        title_tokens = self.tokenizer.tokenize(chapter.title) if with_title else []
        return title_tokens + self.tokenizer.tokenize(chapter.text)

    def get_chapter_sentence_tokens(self, chapter, with_title=True):
        title_sentence_tokens = self.tokenizer.get_sentence_tokens(chapter.title) if with_title else []
        return title_sentence_tokens + self.tokenizer.get_sentence_tokens(chapter.text)

    def get_novel_tokens(self, novel):
        return sum([self.get_chapter_tokens(chapter=chapter) for chapter in tqdm(novel.chapters, desc=novel.title)], [])

    def get_novel_sentence_tokens(self, novel):
        return sum([self.get_chapter_sentence_tokens(chapter) for chapter in tqdm(novel.chapters, desc=novel.title)], [])

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

    def get_novels_sentence_tokens(self, novels):
        return sum([self.get_novel_sentence_tokens(novel=novel) for novel in novels], [])

