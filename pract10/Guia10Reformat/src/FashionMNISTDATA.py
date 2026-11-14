import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FashionMNISTData:
    CLASS_NAMES = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    def __init__(self, batch_size=100, seed=42, color="Reds"):
        self.batch_size = batch_size
        self.seed = seed
        self.color = color

        self.transform = transforms.ToTensor()

        self._load_datasets()
        self._create_loaders()

    def _load_datasets(self):
        full_train_set = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=self.transform
        )

        self.test_set = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=self.transform
        )

        generator = torch.Generator().manual_seed(self.seed)

        self.train_set, self.valid_set = random_split(
            full_train_set,
            [50_000, 10_000],
            generator=generator
        )

        print("\nCantidad de imágenes:")
        print("Train:", len(self.train_set))
        print("Validation:", len(self.valid_set))
        print("Test:", len(self.test_set))

    def _create_loaders(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False
        )

    def inspect_example(self):
        image, label = self.train_set[0]

        print("\nEjemplo:")
        print("Shape:", image.shape)
        print("Etiqueta:", self.CLASS_NAMES[label])
        print("Mínimo:", image.min().item())
        print("Máximo:", image.max().item())

    def show_samples(self, n=6):
        generator = torch.Generator().manual_seed(self.seed)

        indices = torch.randperm(
            len(self.train_set),
            generator=generator
        )[:n].tolist()

        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(7, 7),
            squeeze=False
        )
        print(f"muestra tamano {n}, labels: ")
        for ax, idx in zip(axes.flat, indices):
            _, label = self.train_set[idx]
            print(f"{label}")
