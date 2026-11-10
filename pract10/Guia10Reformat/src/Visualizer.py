import torch
from matplotlib import pyplot as plt


class Visualizer:

    @staticmethod
    def show_reconstructions(
            model,
            data_loader,
            device,
            class_names=None,
            n=6,
            title="Reconstrucciones",
            color="Reds"
    ):

        model.eval()

        images, labels = next(iter(data_loader))

        images = images[:n].to(device)
        labels = labels[:n]

        with torch.no_grad():
            reconstructions = model(images)

        images = images.cpu()
        reconstructions = reconstructions.cpu()

        fig, axes = plt.subplots(
            2,
            n,
            figsize=(2 * n, 4),
            squeeze=False
        )

        for i in range(n):

            axes[0, i].imshow(
                images[i].squeeze(),
                cmap=color
            )

            if class_names is not None:
                axes[0, i].set_title(
                    class_names[labels[i].item()],
                    fontsize=9
                )

            axes[0, i].axis("off")

            axes[1, i].imshow(
                reconstructions[i].squeeze(),
                cmap=color
            )

            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_history(history):

        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(7, 4))

        plt.plot(
            epochs,
            history["train_loss"],
            marker="o",
            label="Train"
        )

        plt.plot(
            epochs,
            history["valid_loss"],
            marker="o",
            label="Validation"
        )

        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.title("Evolución del entrenamiento")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()
