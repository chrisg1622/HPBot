
class Novel:

    def __init__(self, title, chapters):
        self.title = title
        self.chapters = chapters

    def __repr__(self):
        return "{name}(title: '{title}', chapters: '{chapters}')".format(name=self.__class__.__name__, title=self.title, chapters=self.chapters)
