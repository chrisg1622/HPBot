from prefect import Flow as PrefectFlow
from prefect.utilities.debug import raise_on_exception
from prefect.utilities.notifications import slack_notifier


class Flow(PrefectFlow):

    def __init__(self, name, schedule=None):
        super().__init__(name=name, schedule=schedule, state_handlers=[slack_notifier])

    def run(self, *args, **kwargs):
        with raise_on_exception():
            super().run(*args, **kwargs)
