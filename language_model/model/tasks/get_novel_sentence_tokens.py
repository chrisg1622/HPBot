import glob
import json
from language_model.model.tasks.task import Task


class GetNovelSentenceTokens(Task):

    def __init__(self, base_directory, novel_retriever, tokenizer):
        super().__init__()
        self.base_directory = base_directory
        self.novel_retriever = novel_retriever
        self.tokenizer = tokenizer

    def run(self):
        sentence_tokens = []
        for filePath in sorted(glob.glob(pathname=f'{self.base_directory}/*.txt')):
            novel = self.novel_retriever.get_novel(filePath=filePath)
            self.log(f'Processing {novel.title}')
            sentence_tokens.append(self.tokenizer.get_tokens(text=novel.title))
            for chapter in novel.chapters:
                sentence_tokens.append(self.tokenizer.get_tokens(text=chapter.title))
                sentence_tokens.extend(self.tokenizer.get_sentence_tokens(text=chapter.text))
        vocab = sorted({x for sentence in sentence_tokens for x in sentence})
        self.log(f'Saving {len(sentence_tokens)} with a total of {len(vocab)} words')
        json.dump(vocab, open(f'{self.base_directory}/vocabulary.json', 'w'))
        json.dump(sentence_tokens, open(f'{self.base_directory}/all_novels_sentence_tokens.json', 'w'))
        return sentence_tokens
