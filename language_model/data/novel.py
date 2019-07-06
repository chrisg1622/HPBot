
class Novel:

    def __init__(self, title, chapters):
        self.title = title
        self.chapters = chapters

    def __repr__(self):
        return f"{self.__class__.__name_}(title: '{self.title}', chapters: '{self.chapters}')"
