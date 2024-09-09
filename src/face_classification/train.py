import argparse
from pathlib import Path

from clearml import Task
from omegaconf import OmegaConf

from face_classification.trainer import ClassificationTrainer


def main(opt):
    exp_folder = Path(opt.exp_folder)
    config_file_path = exp_folder / "train.yaml"
    if not config_file_path.exists():
        raise ValueError(
            f"The config file for the train should be in the specified exp-folder path and be named `train.yaml`"
        )

    config = OmegaConf.load(config_file_path)

    tags = [
        "classification",
        str(config.MODEL.NAME).replace("_", " ").title(),
        str(config.TRAINING.LOSS.NAME).replace("_", " ").title(),
    ]
    task = Task.init(project_name=config.PROJECT_NAME, task_name=str(exp_folder), tags=tags)
    task.connect_configuration(config_file_path)

    trainer = ClassificationTrainer(config=config, exp_folder=exp_folder)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifiaction model")
    parser.add_argument("--exp-folder", "-e", type=str, help="Directory of the exp")
    opt = parser.parse_args()
    main(opt)
