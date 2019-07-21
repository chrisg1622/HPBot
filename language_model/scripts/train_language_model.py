import fire
import sys
sys.path.insert(0, '../../')
from language_model.model.tasks.train_language_model import TrainLanguageModel


def main(base_directory, model_name, restore_model=True):
    trainLanguageModel = TrainLanguageModel(
        base_directory=base_directory,
        model_name=model_name,
        hidden_size=512,
        restore_model=restore_model
    )
    trainLanguageModel.run(
        batch_size=128,
        learning_rate=0.0001,
        epochs=30,
        batches_per_epoch=100
    )


if __name__ == '__main__':
    fire.Fire(main)
