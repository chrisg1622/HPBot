import logging


class Logger(object):

    def __init__(self, name='logger', file_path=None, level=logging.INFO):
        self.name = name
        self.file_path = file_path
        self.level = level

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter(fmt=f'[%(asctime)s] - {self.name} - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if file_path is not None:
            fh = logging.FileHandler(f'{file_path}', 'w')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
