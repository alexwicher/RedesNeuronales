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

    CLASS_NAMES_V2 = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    def __init__(self, batch_size=100, seed=42, color="Reds",show_dataset_info=True):

        self.train_loader = None
        self.batch_size = batch_size
        self.seed = seed
        self.color = color
        self.split_generator = torch.Generator().manual_seed(self.seed)
        self.train_generator = torch.Generator().manual_seed(self.seed)
        # self.transform = transforms.Normalize((0.286,), (0.353,))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        self._load_datasets(show_dataset_info)
        self._create_loaders()

    def reset_generator(self):
        # Shut down previous workers
        # if self.train_loader._iterator is not None:
        #     self.train_loader._iterator._shutdown_workers()
        #     self.train_loader._iterator = None

        self.train_generator.manual_seed(self.seed)

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            generator=self.train_generator
        )

    def _load_datasets(self,show_dataset_info = True):

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


        self.train_set, self.valid_set = random_split(
            full_train_set,
            [50_000, 10_000],
            generator= self.split_generator
        )
        if show_dataset_info:
            print("\nCantidad de imágenes:")
            print("Train:", len(self.train_set))
            print("Validation:", len(self.valid_set))
            print("Test:", len(self.test_set))

    def _create_loaders(self):
        loader_args = {
            "batch_size": self.batch_size,
            "num_workers": 4, #4 Para mi PC esto es lo mas optimo RTX 4050 // 3th Gen Intel(R) Core(TM) i5-13420H // 32GB DDR4
            "pin_memory": True, # prevents the operating system from paging those particular memory pages to disk.
            "persistent_workers": True
        }

        self.train_loader = DataLoader(
            self.train_set,
            shuffle=True,
            generator=self.train_generator,
            **loader_args
        )

        self.valid_loader = DataLoader(
            self.valid_set,
            shuffle=False,
            **loader_args
        )

        self.test_loader = DataLoader(
            self.test_set,
            shuffle=False,
            **loader_args
        )

    def inspect_example(self):
        image, label = self.train_set[0]
        print("\nEjemplo:")
        print("Shape:", image.shape)
        print("Etiqueta:", self.CLASS_NAMES[label])
        print("Mínimo:", image.min().item())
        print("Máximo:", image.max().item())

    def get_first_batches(self, n_batches=3):
        batches = []

        for i, (_, labels) in enumerate(self.train_loader):
            batches.append(labels.clone())

            if i + 1 == n_batches:
                break

        return batches


