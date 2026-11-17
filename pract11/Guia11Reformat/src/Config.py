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
            lr=0.001,
            seed=42,
            neuron1=128,
            neuron2=64,

            show_init=False,
            show_sample=False,
            show_dimentions=False,
            show_dimentions2=False,
            show_untrained=False,
            show_loss=False,
            show_trained=False,
            show_epoch_data=False,
            show_dataset_info=False
    ):
        self.DROPOUT_P = dropout_p
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.OPTIMIZER_NAME = optimizer_name
        self.LEARNING_RATE = lr
        self.COLOR = color

        self.SEED = seed

        self.neuron_1 = neuron1
        self.neuron_2 = neuron2

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

        if torch.cuda.is_available():  # I had to reduce python 3.12.10 version to get the torch cuda implementations
            torch.cuda.manual_seed_all(self.SEED)
        # setting seeds alone doesn't guarantee that every GPU operation is deterministic.

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # device = torch.device("cpu")
        if self.show_init:

            print("\n========== CONFIGURATION ==========")

            print(f"Seed:             {self.SEED}")
            print(f"Batch size:       {self.BATCH_SIZE}")
            print(f"Epochs:           {self.EPOCHS}")
            print(f"Dropout:          {self.DROPOUT_P}")
            print(f"Optimizer:        {self.OPTIMIZER_NAME}")
            print(f"Learning rate:    {self.LEARNING_RATE}")
            print(f"neuron_1:        {self.neuron_1}")
            print(f"neuron_2:        {self.neuron_2}")
            # print(f"Color:            {self.COLOR}")

            print("\n========== ENVIRONMENT ==========")

            print("PyTorch:", torch.__version__)
            print("Device:", device)

            if device.type == "cuda":
                print("GPU:", torch.cuda.get_device_name(device))

            print("==================================\n")

        return device
