from tqdm.auto import tqdm


class NovelSequenceGenerator:

    OOV_CHAR = '<OOV>'
    START_CHAR = '<START>'
    END_CHAR = '<END>'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_chapter_ngrams(self, chapter, n):
        tokens = self.get_chapter_tokens(chapter=chapter)
        input_list = [self.START_CHAR] * (n - 1) + tokens + [self.END_CHAR]
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_novel_ngrams(self, novel, n):
        return [
            ngram
            for chapter in tqdm(novel.chapters)
            for ngram in self.get_chapter_ngrams(chapter=chapter, n=n)
        ]

    def get_chapter_tokens(self, chapter):
        return self.tokenizer.tokenize(chapter.text)

    def get_novel_tokens(self, novel):
        return [token for chapter in tqdm(novel.chapters) for token in self.get_chapter_tokens(chapter=chapter)]

    def get_novel_vocabulary(self, novel):
        return [self.OOV_CHAR, self.START_CHAR, self.END_CHAR] + sorted(set(self.get_novel_tokens(novel=novel)))
