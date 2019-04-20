import spacy


class SpacyTokenizer:

    def __init__(self):
        self.tagger = spacy.load('en_core_web_sm')

    def tokenize(self, text):
        return [token.text for token in self.tagger(text)]
