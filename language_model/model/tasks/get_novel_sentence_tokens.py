import glob
import json
from language_model.data.constants import START_TOKEN
from language_model.data.constants import EOS_TOKEN
from language_model.model.tasks.task import Task


class GetNovelSentenceTokens(Task):

    TRAINING_SEQUENCES_FILE_NAME = 'all_novels_sentence_tokens.json'
    VOCABULARY_FILE_NAME = 'vocabulary.json'

    def __init__(self, base_directory, novel_retriever, tokenizer):
        super().__init__(base_directory=base_directory)
        self.novel_retriever = novel_retriever
        self.tokenizer = tokenizer

    @staticmethod
    def pad_tokens(tokens):
        return [START_TOKEN] + tokens + [EOS_TOKEN]

    def _run(self):
        sentence_tokens = []
        for filePath in sorted(glob.glob(pathname=f'{self.base_directory}/*.txt')):
            novel = self.novel_retriever.get_novel(filePath=filePath)
            self.log(f'Processing {novel.title}')
            sentence_tokens.append(self.tokenizer.get_tokens(text=novel.title))
            for chapter in novel.chapters:
                sentence_tokens.append(self.pad_tokens(tokens=self.tokenizer.get_tokens(text=chapter.title)))
                for tokens in self.tokenizer.get_sentence_tokens(text=chapter.text):
                    padded_tokens = self.pad_tokens(tokens=tokens)
                    sentence_tokens.append(padded_tokens)
        vocab = sorted({x for sentence in sentence_tokens for x in sentence})
        self.log(f'Saving {len(sentence_tokens)} with a total of {len(vocab)} words')
        json.dump(vocab, open(f'{self.base_directory}/{self.VOCABULARY_FILE_NAME}', 'w'))
        json.dump(sentence_tokens, open(f'{self.base_directory}/{self.TRAINING_SEQUENCES_FILE_NAME}', 'w'))
        return sentence_tokens
