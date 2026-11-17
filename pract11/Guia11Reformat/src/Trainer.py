import torch

from pract11.Guia11Reformat.src.FashionMNISTData import FashionMNISTData


class Trainer:

    def __init__(self, model, optimizer, loss_fn, device, config):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config

        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "test_loss": [],
            "train_acu": [],
            "valid_acu": [],
            "test_acu": []
        }

    def train_one_epoch(self, data_loader):

        self.model.train()

        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device)
        total_examples = 0

        for images, labels in data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            batch_size = labels.size(0)

            predictions = logits.argmax(dim=1)

            total_loss += loss.detach() * batch_size  # The loss participates in backpropagation.
            # But afterward, we want to accumulate its numerical value to calculate the average training loss.
            # We don't need gradients for that operation.therefor .detach()
            total_correct += (predictions == labels).sum()
            total_examples += batch_size

        if total_examples == 0:
            raise ValueError("The data_loader is empty.")

        avg_loss = (total_loss / total_examples).item()
        accuracy = (total_correct / total_examples).item()

        return avg_loss, accuracy

    # It also disables gradient tracking, but additionally disables some autograd-related bookkeeping,
    # such as view tracking and version-counter updates.
    @torch.inference_mode()  # Both torch.no_grad() and torch.inference_mode() tell PyTorch not to track operations for
    # gradient computation.
    # They are used when you don't need backpropagation, such as during validation, testing, and inference.
    def evaluate(self, data_loader, do_acc_per_class=False, do_confusion_matrix=False):

        self.model.eval()

        num_classes = len(FashionMNISTData.CLASS_NAMES)

        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device)
        total_examples = 0

        # Per-class counters
        class_correct = torch.zeros(num_classes, device=self.device)
        class_total = torch.zeros(num_classes, device=self.device)

        # Confusion matrix
        confusion_matrix = torch.zeros(
            num_classes, num_classes, device=self.device
        )

        for images, labels in data_loader:

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss * batch_size
            total_correct += (predictions == labels).sum()
            total_examples += batch_size

            if do_acc_per_class:
                class_total += torch.bincount(  # counts how many times each integer value appears in a tensor.
                    labels, minlength=num_classes
                )

                class_correct += torch.bincount(
                    labels[predictions == labels],
                    minlength=num_classes
                )

            if do_confusion_matrix:
                batch_cm = torch.bincount(
                    labels * num_classes + predictions,
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)

                confusion_matrix += batch_cm

        # Global metrics
        avg_loss = (total_loss / total_examples).item()
        accuracy = (total_correct / total_examples).item()

        # Per-class accuracy
        acc_per_class = {}

        if do_acc_per_class:
            for label in range(num_classes):
                if class_total[label] > 0:
                    acc_per_class[label] = (
                            class_correct[label] / class_total[label]
                    ).item()
                else:
                    acc_per_class[label] = float("nan")

        return (
            avg_loss,
            accuracy,
            acc_per_class,
            confusion_matrix.cpu()
        )

    def fit(self, train_loader, valid_loader, epochs=10):

        print(
            f"\nIniciando entrenamiento... "
            f"Opti -> {self.config.OPTIMIZER_NAME} "
            f"| Drop -> {self.config.DROPOUT_P} "
            f"| Batch size -> {self.config.BATCH_SIZE}"
        )

        for epoch in range(1, epochs + 1):

            # Training
            train_loss, train_acu = self.train_one_epoch(train_loader)

            # Validation
            valid_loss, valid_acu, _, _ = self.evaluate(valid_loader)

            # History
            self.history["train_loss"].append(train_loss)
            self.history["valid_loss"].append(valid_loss)
            self.history["train_acu"].append(train_acu)
            self.history["valid_acu"].append(valid_acu)

            if self.config.show_epoch_data:
                print(
                    f"Epoch {epoch:02d}/{epochs} | "
                    f"Train CEL: {train_loss:.5f} | "
                    f"Valid CEL: {valid_loss:.5f} | "
                    f"Train Accuracy: {train_acu:.5f} | "
                    f"Valid Accuracy: {valid_acu:.5f}"
                )

    def test(self, test_loader):

        test_loss, test_acu, acc_per_class, confusion_matrix = self.evaluate(
            test_loader,
            do_acc_per_class=True,
            do_confusion_matrix=True
        )

        self.history["test_loss"].append(test_loss)
        self.history["test_acu"].append(test_acu)

        print(f"\nTest CEL: {test_loss:.5f}")
        print(f"Test accuracy: {test_acu * 100:.2f}%\n")

        return test_loss, test_acu, confusion_matrix
