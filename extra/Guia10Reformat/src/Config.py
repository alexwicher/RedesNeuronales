import random

import numpy as np
import torch


class Config:
    SEED = 42

    SGD_LR = 0.01
    ADAM_LR = 0.001

    def __init__(
            self,
            latent_dim=64,
            dropout_p=0.2,
            batch_size=100,
            epochs=10,
            optimizer_name="Adam",
            color="Reds",

            showSample=True,
            showDimentions=True,
            showDimentions2=True,
            showUntrained=True,
            showLoss=True,
            showTrained=True,
    ):
        self.LATENT_DIM = latent_dim
        self.DROPOUT_P = dropout_p
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.OPTIMIZER_NAME = optimizer_name
        self.COLOR = color

        self.showSample = showSample
        self.showDimentions = showDimentions
        self.showDimentions2 = showDimentions2
        self.showUntrained = showUntrained
        self.showLoss = showLoss
        self.showTrained = showTrained

    def setup(self):

        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("\n========== CONFIGURATION ==========")

        print(f"Seed:             {Config.SEED}")
        print(f"Batch size:       {self.BATCH_SIZE}")
        print(f"Epochs:           {self.EPOCHS}")
        print(f"Latent dimension: {self.LATENT_DIM}")
        print(f"Dropout:          {self.DROPOUT_P}")
        print(f"Optimizer:        {self.OPTIMIZER_NAME}")
        # print(f"Color:            {self.COLOR}")

        print("\n========== ENVIRONMENT ==========")

        print("PyTorch:", torch.__version__)
        print("Device:", device)

        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(device))

        print("==================================\n")

        return device
