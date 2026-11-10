import torch


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
            "train_acu": [],
            "valid_acu": []
        }

    def train_one_epoch(self, data_loader):

        self.model.train()

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.loss_fn(logits, labels)

            loss.backward()

            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, data_loader):

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)

            loss = self.loss_fn(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += torch.sum(predictions == labels).item()
            total_examples += batch_size

        return total_loss / total_examples, total_correct / total_examples

    def fit(self, train_loader, valid_loader, epochs=10):

        print(
            f"\nIniciando entrenamiento... Opti -> {self.config.OPTIMIZER_NAME} | Drop -> {self.config.DROPOUT_P} "
            f"| Batch size -> {self.config.BATCH_SIZE}")

        for epoch in range(1, epochs + 1):
            self.train_one_epoch(train_loader)

            train_loss, train_acu = self.evaluate(train_loader)

            valid_loss, valid_acu = self.evaluate(valid_loader)

            self.history["train_loss"].append(train_loss)
            self.history["valid_loss"].append(valid_loss)
            self.history["train_acu"].append(train_acu)
            self.history["valid_acu"].append(valid_acu)

            if self.config.showEpochData:
                print(
                    f"Época {epoch:02d}/{epochs} | "
                    f"Train CEL: {train_loss:.5f} | "
                    f"Valid CEL: {valid_loss:.5f} | "
                    f"Train Accuracy: {train_acu:.5f} | "
                    f"Valid Accuracy: {valid_acu:.5f}"
                )

    def test(self, test_loader):

        test_loss, test_acu = self.evaluate(test_loader)

        print(f"\nTest CEL: {test_loss:.5f}")
        print(f"\nTest accuracy: {test_acu:.5f}")

        return test_loss, test_acu
