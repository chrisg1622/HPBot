import spacy
import os


class SpacyTokenizer:

    def __init__(self):
        self.tagger = spacy.load(os.environ['SPACY_PATH'])

    def get_tokens(self, text):
        return [token.text for token in self.tagger(text)]

    def get_sentence_tokens(self, text):
        return [[token.text for token in sent] for sent in self.tagger(text).sents]
