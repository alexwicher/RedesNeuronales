import torch


def create_optimizer(model, optimizer_name, adam_lr, sgd_lr):
    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=sgd_lr
        )

    elif optimizer_name == "Adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=adam_lr
        )
    else:
        raise ValueError(
            "Optimizador no reconocido."
        )