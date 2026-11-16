import seaborn as sns
import torch
from matplotlib import pyplot as plt

from pract11.Guia11Reformat.src.FashionMNISTData import FashionMNISTData


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
        plots = [
            {
                "metrics": ["train_loss", "valid_loss"],
                "labels": ["Train loss", "Validation loss"],
                "ylabel": "CEL",
                "xlabel": "Epoch",
                "linestyles": ["--", "-"]
            },
            {
                "metrics": ["train_acu", "valid_acu"],
                "labels": ["Train accuracy", "Validation accuracy"],
                "ylabel": "Accuracy",
                "xlabel": "Epoch",
                "linestyles": ["--", "-"]
            },
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, plot in zip(axes, plots):
            for exp_title, item in results.items():
                history = item["trainer"].history

                for metric, label, linestyle in zip(
                        plot["metrics"],
                        plot["labels"],
                        plot["linestyles"]
                ):
                    epochs = range(1, len(history[metric]) + 1)

                    ax.plot(
                        epochs,
                        history[metric],
                        marker="o",
                        linestyle=linestyle,
                        label=f"{label} {exp_title}"
                    )

            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        for exp_title, item in results.items():
            trainer = item["trainer"]
            data = item["data"]
            confusion_matrix = item["confusion_matrix"]

            # show_inaccuracies(trainer, data)
            show_confusion_matrix(confusion_matrix, exp_title)


def show_inaccuracies(
        trainer, data, n=6,
        color="Reds"
):
    model = trainer.model
    data_loader = data.test_loader
    model.eval()
    images, labels = next(iter(data_loader))

    images = images.to(trainer.device)
    labels = labels.to(trainer.device)

    with torch.no_grad():
        logits = model(images)

    predictions = logits.argmax(dim=1)

    incorrect_mask = predictions != labels
    incorrect_images = images[incorrect_mask][:n]
    incorrect_labels = labels[incorrect_mask][:n]
    incorrect_predictions = predictions[incorrect_mask][:n]

    incorrect_images = incorrect_images * 0.5 + 0.5  # Denormalization

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(2 * n, 4),
        squeeze=False
    )
    for i in range(n):
        img = incorrect_images[i].cpu().permute(1, 2, 0)
        axes[0, i].imshow(
            img.squeeze(),
            cmap=color
        )

        axes[0, i].set_title(
            f"True: {FashionMNISTData.CLASS_NAMES[incorrect_labels[i].item()]} \n"
            f"Predicted: {FashionMNISTData.CLASS_NAMES[incorrect_predictions[i].item()]}",
            fontsize=9
        )
        axes[0, i].axis("off")

    plt.tight_layout()
    plt.show()


def show_confusion_matrix(confusion_matrix, title):
    cm = confusion_matrix.cpu()
    cm = cm.float() / cm.sum(dim=1, keepdim=True) #normalized
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=FashionMNISTData.CLASS_NAMES_V2,
        yticklabels=FashionMNISTData.CLASS_NAMES_V2
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix {title}")

    plt.tight_layout()
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
