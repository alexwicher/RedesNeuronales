import torch


class Trainer:

    def __init__(self, model, optimizer, loss_fn, device):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.history = {
            "train_loss": [],
            "valid_loss": []
        }

    def train_one_epoch(self, data_loader):

        self.model.train()

        for images, _ in data_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()

            reconstructions = self.model(images)

            loss = self.loss_fn(reconstructions, images)

            loss.backward()

            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, data_loader):

        self.model.eval()

        total_loss = 0.0
        total_examples = 0

        for images, _ in data_loader:
            images = images.to(self.device)

            reconstructions = self.model(images)

            loss = self.loss_fn(reconstructions, images)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / total_examples

    def fit(self, train_loader, valid_loader, epochs=10):

        print("\nIniciando entrenamiento...")
        print("Dispositivo:", self.device)

        for epoch in range(1, epochs + 1):
            self.train_one_epoch(train_loader)

            train_loss = self.evaluate(train_loader)

            valid_loss = self.evaluate(valid_loader)

            self.history["train_loss"].append(train_loss)
            self.history["valid_loss"].append(valid_loss)

            print(
                f"Época {epoch:02d}/{epochs} | "
                f"Train MSE: {train_loss:.5f} | "
                f"Valid MSE: {valid_loss:.5f}"
            )

    def test(self, test_loader):

        test_loss = self.evaluate(test_loader)

        print(f"\nTest MSE: {test_loss:.5f}")

        return test_loss
