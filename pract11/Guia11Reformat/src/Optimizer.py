import torch


def create_optimizer(model, optimizer_name, lr):
    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr
        )

    elif optimizer_name == "Adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
    else:
        raise ValueError(
            "Optimizador no reconocido."
        )