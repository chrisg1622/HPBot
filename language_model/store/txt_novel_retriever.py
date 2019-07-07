from language_model.data.chapter import Chapter
from language_model.data.novel import Novel


class TxtNovelRetriever:

    @staticmethod
    def get_novel(filePath):
        lines = []
        with open(filePath, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').strip('\x0c')
                if line != '' and not(line.isdigit()):
                    lines.append(line)
        novel_title = lines[0]
        chapters = []
        title = ''
        text = ''
        for lineIndex, line in enumerate(lines[1:], 1):
            if line.isupper():
                if line.startswith('CHAPTER'):
                    if line not in title:
                        if title != '':
                            chapters.append(Chapter(title=title, text=text))
                        title = line + ' -' * int(' -' not in line)
                        text = ''
                elif lines[lineIndex - 1] in title and 50 > len(line) > 1 and '“' not in line:
                    if line not in title:
                        title += ' ' + line
                else:
                    text += ' ' * int(len(text) > 2) + line
            else:
                text += ' ' * int(len(text) > 2) + line
        chapters.append(Chapter(title=title, text=text))
        return Novel(title=novel_title, chapters=chapters)
