import torch

from src.Config import Config


def create_optimizer(model, optimizer_name):

    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=Config.SGD_LR
        )

    elif optimizer_name == "Adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=Config.ADAM_LR
        )
    else:
        raise ValueError(
            "Optimizador no reconocido."
        )