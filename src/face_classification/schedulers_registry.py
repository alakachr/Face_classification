import torch

SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
}
