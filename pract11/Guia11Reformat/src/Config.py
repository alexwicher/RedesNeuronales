import random

import numpy as np
import torch


class Config:
    def __init__(
            self,
            dropout_p=0.2,
            batch_size=100,
            epochs=10,
            optimizer_name="Adam",
            color="Reds",
            sgd_lr=0.01,
            adam_lr=0.001,
            seed=42,

            show_init=True,
            show_sample=True,
            show_dimentions=True,
            show_dimentions2=True,
            show_untrained=True,
            show_loss=True,
            show_trained=True,
            show_epoch_data=True,
            show_dataset_info=True
    ):
        self.DROPOUT_P = dropout_p
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.OPTIMIZER_NAME = optimizer_name
        self.COLOR = color

        self.SEED = seed
        self.ADAM_LR = adam_lr
        self.SGD_LR = sgd_lr

        self.show_dataset_info = show_dataset_info
        self.show_init = show_init
        self.show_sample = show_sample
        self.show_dimentions = show_dimentions
        self.show_dimentions2 = show_dimentions2
        self.show_untrained = show_untrained
        self.show_loss = show_loss
        self.show_trained = show_trained
        self.show_epoch_data = show_epoch_data

    def setup(self):

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.show_init:

            print("\n========== CONFIGURATION ==========")

            print(f"Seed:             {self.SEED}")
            print(f"Batch size:       {self.BATCH_SIZE}")
            print(f"Epochs:           {self.EPOCHS}")
            print(f"Dropout:          {self.DROPOUT_P}")
            print(f"Optimizer:        {self.OPTIMIZER_NAME}")
            print(f"Adam_LR:        {self.ADAM_LR}")
            print(f"SGD_LR:        {self.SGD_LR}")
            # print(f"Color:            {self.COLOR}")

            print("\n========== ENVIRONMENT ==========")

            print("PyTorch:", torch.__version__)
            print("Device:", device)

            if device.type == "cuda":
                print("GPU:", torch.cuda.get_device_name(device))

            print("==================================\n")

        return device
