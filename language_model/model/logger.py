import logging


class Logger(object):

    def __init__(self, name='logger', file_path=None, level=logging.INFO):
        self.name = name
        self.file_path = file_path
        self.level = level

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if file_path is not None:
            fh = logging.FileHandler(f'{file_path}', 'w')
            self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
