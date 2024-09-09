import torch

OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW,
}
