from pathlib import Path
from typing import Any, Literal

import torch
from albumentations import Compose
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import MetricCollection
from tqdm import tqdm

from face_classification.augmentations import build_augment_compose
from face_classification.dataset import FaceDataset
from face_classification.losses_registry import LOSSES
from face_classification.metrics_registry import CLASSIFICATION_METRICS
from face_classification.models_registry import CLASSIFICATION_MODELS
from face_classification.optimizers_registry import OPTIMIZERS
from face_classification.schedulers_registry import SCHEDULERS
from face_classification.settings import logging


class ClassificationTrainer:
    def __init__(self, config: DictConfig, exp_folder: Path, *args, **kwargs) -> None:
        self.config = config
        self.exp_folder = exp_folder

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logging.info(f"Using device {self.device}")

        if config.UTILS.ENABLE_TENSORBOARD:
            writer_path = self.exp_folder / "tensorboard"
            if not writer_path.exists():
                writer_path.mkdir()
            self.writer = SummaryWriter(log_dir=writer_path)

        self.best_validation_metric = 0
        self.lr_scheduler = None

    def build_model(self) -> torch.nn.Module:
        model = CLASSIFICATION_MODELS[self.config.MODEL.NAME](**{k: v for k, v in self.config.MODEL.PARAMETERS.items()})
        model.to(self.device)

        return model

    def _get_data_loader(self, partition: Literal["train", "val", "test"]) -> DataLoader:

        do_augmentation = partition == "train"
        augmentations: Compose | None = (
            build_augment_compose(self.config.AUGMENTATIONS) if do_augmentation and self.config.AUGMENTATIONS else None
        )
        partition2parameters_key = {"train": "TRAIN_DATASET_PARAMETERS", "val": "VAL_DATASET_PARAMETERS"}
        dataset = FaceDataset(
            **{k.lower(): v for k, v in self.config.DATA[partition2parameters_key[partition]].items()},
            partition=partition,
            do_augmentation=do_augmentation,
            augmentations=augmentations,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.DATA.BATCH_SIZE,
            shuffle=True if partition == "train" else False,
            drop_last=True,
            num_workers=self.config.UTILS.NUM_WORKERS,
        )
        logging.info(f"\tLength of {partition} dataset: {len(dataset)}")
        return dataloader

    def create_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        logging.info("Creating data loaders..")
        train_data_loader = self._get_data_loader("train")
        val_data_loader = self._get_data_loader("val")

        logging.info("\tData loaders created")

        return train_data_loader, val_data_loader, None

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = OPTIMIZERS[self.config.TRAINING.OPTIMIZER.NAME](
            params=model.parameters(),
            **{k.lower(): v for k, v in self.config.TRAINING.OPTIMIZER.PARAMETERS.items()},
        )
        return optimizer

    def initialize_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
        if self.config.TRAINING.SCHEDULER.USE:
            logging.info("Initializing learning rate scheduler..")
            lr_scheduler = SCHEDULERS[self.config.TRAINING.SCHEDULER.NAME](
                optimizer=optimizer,
                **{k: v for k, v in self.config.TRAINING.SCHEDULER.PARAMETERS.items()},
            )
            logging.info("\tLearning rate scheduler initialized")
        else:
            logging.info("No learning rate scheduler selected")
            lr_scheduler = None
        return lr_scheduler

    def build_model_optimizer(self) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
        logging.info("Building model & optimizer..")
        start_epoch = 0
        model = self.build_model()

        nparameters = sum(p.numel() for p in model.parameters())
        logging.info("\tModel built")
        logging.info(f"\tModel has {nparameters} parameters")

        optimizer = self.build_optimizer(model)
        logging.info("\tOptimizer built")

        return model, optimizer, start_epoch

    def initialize_criterion(self) -> torch.nn.Module:
        logging.info("Initializing loss..")
        if self.config.TRAINING.LOSS.PARAMETERS:
            criterion_parameters = {k.lower(): v for k, v in self.config.TRAINING.LOSS.PARAMETERS.items()}
            criterion = LOSSES[self.config.TRAINING.LOSS.NAME](**criterion_parameters)
        else:
            criterion = LOSSES[self.config.TRAINING.LOSS.NAME]()
        logging.info("\tLoss initialized")
        return criterion

    def initialize_metrics(self) -> MetricCollection:
        metrics = {}
        for metric in self.config.EVALUATION.METRICS:
            metrics[metric.DISPLAY_NAME] = CLASSIFICATION_METRICS[metric.NAME](**metric.PARAMETERS)
        metrics_collection = MetricCollection(metrics=metrics).to(self.device)
        return metrics_collection

    def train(self):
        logging.info("Creating data loaders..")
        train_data_loader, val_data_loader, _ = self.create_data_loaders()
        model, optimizer, start_epoch = self.build_model_optimizer()
        criterion = self.initialize_criterion()
        epoch_lr_scheduler = self.initialize_scheduler(optimizer=optimizer)

        logging.info("Training model..")
        n_epochs = self.config.TRAINING.MAX_EPOCHS
        print_every_batch = self.config.UTILS.LOGGING_ITERATION_FREQ

        for epoch in range(start_epoch + 1, n_epochs + 1):
            loss, training_metrics = self.train_one_epoch(
                train_data_loader=train_data_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                log_period=print_every_batch,
                epoch=epoch,
                tensorboard_writer=self.writer,
            )

            if self.writer:
                self.writer.add_scalar("Total Loss/train", loss, epoch)
                for name, value in training_metrics.items():
                    if value.shape == torch.Size([]):
                        self.writer.add_scalar(f"{name}/train", value.item(), epoch)
                    else:
                        for index, v in enumerate(value):
                            self.writer.add_scalar(f"{name}/train/{index}", v.item(), epoch)

            if (epoch % self.config.UTILS.EVAL_EPOCH_FREQ == 0) and (epoch > 1):
                loss, validation_metrics = self.validate_one_epoch(
                    val_data_loader=val_data_loader,
                    model=model,
                    criterion=criterion,
                    print_every_batch=print_every_batch,
                    epoch=epoch,
                )

                if self.writer:
                    self.writer.add_scalar("Total Loss/validation", loss, epoch)
                    for name, value in validation_metrics.items():
                        if value.shape == torch.Size([]):
                            self.writer.add_scalar(f"{name}/validation", value.item(), epoch)
                        else:
                            for index, v in enumerate(value):
                                self.writer.add_scalar(f"{name}/validation/{index}", v.item(), epoch)

                self.save(
                    epoch=epoch,
                    model=model,
                    opt=optimizer,
                    train_metrics=training_metrics,
                    val_metrics=validation_metrics,
                    classes=self.config.DATA.CLASSES,
                )

            if epoch_lr_scheduler:
                epoch_lr_scheduler.step()

    def train_one_epoch(
        self,
        train_data_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        log_period: int,
        epoch: int,
        tensorboard_writer: SummaryWriter | None = None,
    ):
        model.train()

        metrics_collection = self.initialize_metrics()

        model = model.to(self.device)
        iteration = 0
        for images, targets in tqdm(
            train_data_loader,
        ):
            iteration += 1
            images: torch.Tensor = images.to(device=self.device)
            targets: torch.Tensor = targets.to(device=self.device)

            predictions = model(images)
            total_loss = criterion(predictions, targets)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            metrics = metrics_collection(predictions, targets)

            if (iteration % log_period == 0) and (iteration > 0):
                if tensorboard_writer:
                    tensorboard_writer.add_scalar(
                        "Avg loss/training",
                        total_loss,
                        global_step=iteration + (len(train_data_loader)) * (epoch - 1),
                    )
            logging.info(f"Epoch: {epoch}, Iteration: {iteration}, Loss: {total_loss}")

        metrics = metrics_collection.compute()
        logging.info(f"Epoch: {epoch}, Loss: {total_loss}, Metrics: {metrics}")
        metrics_collection.reset()

        if tensorboard_writer:
            tensorboard_writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], global_step=epoch)

        return total_loss, metrics

    def validate_one_epoch(
        self,
        val_data_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        print_every_batch: int,
        epoch: int,
    ):
        model.eval()

        metrics_collection = self.initialize_metrics()

        with torch.inference_mode():
            for images, targets in tqdm(
                val_data_loader,
            ):
                images: torch.Tensor = images.to(device=self.device)
                targets: torch.Tensor = targets.to(device=self.device)
                predictions: torch.Tensor = model(images)

                total_loss = criterion(predictions, targets)

                metrics = metrics_collection(predictions, targets)

        metrics = metrics_collection.compute()
        metrics_collection.reset()

        return total_loss, metrics

    @property
    def checkpoints_path(self) -> Path:
        checkpoints_folder_path = self.exp_folder / "checkpoints"
        checkpoints_folder_path.mkdir(parents=True, exist_ok=True)

        return checkpoints_folder_path

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        classes: list[int],
    ):
        model_state_dict = model.state_dict()
        if torch.cuda.device_count() > 1:  # Multi-GPU
            model_state_dict = model.module.state_dict()

        learning_dict = {
            "epoch": epoch,
            "classes": classes,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": opt.state_dict(),
            "config": self.config,
        }

        save_dict = {
            **learning_dict,
            **{"train_metrics": train_metrics},
            **{"val_metrics": val_metrics},
        }

        self._save(epoch=epoch, save_dict=save_dict, learning_dict=learning_dict, val_metrics=val_metrics)

    def _save(
        self,
        epoch: int,
        save_dict: dict[str, Any],
        learning_dict: dict[str, Any],
        val_metrics: dict[str, Any],
    ):
        save_path = self.checkpoints_path / f"ckpt_{epoch}.pth"

        torch.save(save_dict, save_path)
        logging.info(f"Saved model to: {save_path}")
        latest_save_path = self.checkpoints_path / "ckpt_latest.pth"
        torch.save(learning_dict, latest_save_path)
        logging.info(f"Saved model to: {latest_save_path}")

        if val_metrics:
            current_validation_metric = val_metrics[self.config.EVALUATION.MAIN_METRIC].detach().item()
            difference = current_validation_metric - self.best_validation_metric
            if self.config.EVALUATION.MINIMIZE_MAIN_METRIC:
                difference *= -1
            if difference > 0:
                logging.info(
                    f"Performance improvement +{difference:.4f}, from {self.best_validation_metric:.4f} to {current_validation_metric:.4f}"
                )
                self.best_validation_metric = current_validation_metric
                best_save_path = self.checkpoints_path / "ckpt_best.pth"
                torch.save(save_dict, best_save_path)
                logging.info(f"Saved model to: {best_save_path}")
