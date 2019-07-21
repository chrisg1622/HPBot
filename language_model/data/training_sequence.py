
class TrainingSequence:

    def __init__(self, tokens, target_token):
        self.tokens = tokens
        self.target_token = target_token

    def to_json(self):
        return {'tokens': self.tokens, 'target_token': self.target_token}

    @classmethod
    def from_json(self, json_object):
        return TrainingSequence(tokens=json_object['tokens'], target_token=json_object['target_token'])
