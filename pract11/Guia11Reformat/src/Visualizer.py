import torch
from matplotlib import pyplot as plt


class Visualizer:

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
        plt.ylabel("CEL")
        plt.title("Evolución del entrenamiento")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

        plt.plot(
            epochs,
            history["train_acu"],
            marker="o",
            label="Train accuracy"
        )

        plt.plot(
            epochs,
            history["valid_acu"],
            marker="o",
            label="Validation accuracy"
        )

        plt.xlabel("Época")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

    @staticmethod
    def showPredictions(model, device, data, n=6):
        model.eval()
        images, labels = next(iter(data.train_loader))
        images = images[:n].to(device)
        labels = labels[:n].to(device)
        with torch.no_grad():
            model_images = model(images)
            trained_logits_raw = model_images
            trained_logits_labels = model_images.argmax(dim=1)
            trained_logits_probs = model_images.softmax(dim=1) * 100

        trained_logits_labels = trained_logits_labels.cpu()
        torch.set_printoptions(precision=2, sci_mode=False)
        print(f"TRAINED LOGITS RAW: {trained_logits_raw}")
        print(f"PREDICTED PROBABILITIES (%): {trained_logits_probs}")
        print(f"PREDICTED LABELS: {trained_logits_labels}")
        print(f"ACTUAL LABELS: {labels}")

    @staticmethod
    def showExpieResults(results):
        fig, axs = plt.subplots(1, 2, figsize=(7, 4))
        for title, trainer in results:
            history = trainer.history
            epochs = range(1, len(history["train_loss"]) + 1)

            axs[0].plot(
                epochs,
                history["train_loss"],
                linestyle="--",
                marker="o",
                label="Train"
            )

            axs[0].plot(
                epochs,
                history["valid_loss"],
                marker="o",
                label="Validation"
            )

            axs[0].xlabel("Época")
            axs[0].ylabel("CEL")
            axs[0].title("Evolución del entrenamiento")

            axs[1].plot(
                epochs,
                history["train_acu"],
                marker="o",
                linestyle="--",
                label="Train accuracy"
            )

            axs[1].plot(
                epochs,
                history["valid_acu"],
                marker="o",
                label="Validation accuracy"
            )

            axs[1].xlabel("Época")
            axs[1].ylabel("Accuracy")

        plt.show()