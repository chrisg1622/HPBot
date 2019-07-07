from prefect import Task as PrefectTask
from prefect.utilities.notifications import slack_notifier
from language_model.model.logger import Logger


class Task(PrefectTask):

    def __init__(self, logger_file_path=None):
        super().__init__(name=self.__class__.__name__, state_handlers=[slack_notifier], skip_on_upstream_skip=False)
        self.logger = Logger(name=self.name, file_path=logger_file_path)

    def log(self, message):
        self.logger.info(msg=message)
