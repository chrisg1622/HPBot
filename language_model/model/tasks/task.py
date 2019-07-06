from prefect import Task as PrefectTask
from prefect.utilities.notifications import slack_notifier


class Task(PrefectTask):

    def __init__(self):
        super().__init__(name=self.__class__.__name__, state_handlers=[slack_notifier], skip_on_upstream_skip=False)

