from prefect import Task as PrefectTask
from prefect.utilities.notifications import slack_notifier
from language_model.model.logger import Logger


class Task(PrefectTask):

    def __init__(self, base_directory=None):
        self.base_directory = base_directory
        super().__init__(name=self.__class__.__name__, state_handlers=[slack_notifier], skip_on_upstream_skip=False)
        self.logger = Logger(name=self.name, file_path=f'{base_directory}/{self.name}.log' if base_directory is not None else None)

    def log(self, message):
        self.logger.info(msg=message)
