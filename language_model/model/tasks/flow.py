from prefect import Flow as PrefectFlow
from prefect.utilities.notifications import slack_notifier


class Flow(PrefectFlow):

    def __init__(self, name, schedule=None):
        super().__init__(name=name, schedule=schedule, state_handlers=[slack_notifier])

