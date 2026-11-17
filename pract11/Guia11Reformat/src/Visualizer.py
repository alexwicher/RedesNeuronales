import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

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
    def showExpieResults(results, show_confusion=True):

        plots = [
            {
                "metrics": [("train_acu", "--"), ("valid_acu", "-")],
                "scaleY": 2.3,
                "ylabel": "CEL",
                "xlabel": "Epoch",
            },
            {
                "metrics": [("train_acu", "--"), ("valid_acu", "-")],
                "scaleY": 1,
                "ylabel": "Accuracy",
                "xlabel": "Epoch",
            },
        ]

        # Get Matplotlib's default color palette
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Assign one color per experiment
        for i, (exp_title, item) in enumerate(results.items()):
            print(f"----------------- {exp_title} -----------------")
            history = item["trainer"].history
            color = colors[i % len(colors)]

            # Plot the experiment on both subplots
            for ax, plot in zip(axes, plots):
                for j, (metric, linestyle) in enumerate(plot["metrics"]):
                    epochs = range(1, len(history[metric]) + 1)
                    ax.plot(
                        epochs,
                        history[metric],
                        color=color,
                        marker="o",
                        linestyle=linestyle,
                        label=exp_title if j == 0 else None
                    )

        # Configure subplots
        for ax, plot in zip(axes, plots):
            ax.set_xlabel(plot["xlabel"])
            ax.set_ylabel(plot["ylabel"])
            # ax.set_ylim(0, plot["scaleY"])
            ax.legend()
            ax.grid(alpha=0.3)

        style_legend = [
            Line2D([0], [0], color="gray", linestyle="-", label="Validation"),
            Line2D([0], [0], color="gray", linestyle="--", label="Training")
        ]
        fig.legend(handles=style_legend, loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

        # Show confusion matrices
        for exp_title, item in results.items():
            if show_confusion: show_confusion_matrix(item["confusion_matrix"], exp_title)

    @staticmethod
    def showComparativeTable(results):

        data = {}

        for exp_title, item in results.items():
            trainer = item["trainer"]
            history = trainer.history

            cm = item["confusion_matrix"]
            # Normalize each row
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

            # Support both numpy arrays and PyTorch tensors
            if isinstance(cm, torch.Tensor):
                cm = cm.detach().cpu().numpy()
            cm = np.asarray(cm, dtype=float)

            # Handle percentage-normalized matrices
            if cm.max() > 1:
                cm = cm / 100.0

            # Ideal confusion matrix
            ideal = np.eye(cm.shape[0])

            # Per-class recall (diagonal of row-normalized CM)
            recalls = np.diag(cm)

            # Per-class precision
            column_sums = cm.sum(axis=0)
            precisions = np.divide(
                np.diag(cm),
                column_sums,
                out=np.zeros_like(column_sums),
                where=column_sums != 0
            )

            # Per-class F1
            f1_scores = np.divide(
                2 * precisions * recalls,  # 2 * precisions * recalls / precisions + recalls
                precisions + recalls,
                out=np.zeros_like(recalls),
                where=(precisions + recalls) != 0
            )

            train_acc = history["train_acu"][-1]
            valid_acc = history["valid_acu"][-1]

            # Generalization metrics
            generalization_gap = train_acc - valid_acc
            loss_gap = history["valid_loss"][-1] - history["train_loss"][-1]

            # Confusion matrix deviation (squared Frobenius distance)
            cm_deviation = np.sum((cm - ideal) ** 2)

            data[exp_title] = {
                "Train Loss": history["train_loss"][-1],
                "Valid Loss": history["valid_loss"][-1],
                "Train Accu": train_acc,
                "Valid Accu": valid_acc,
                "Test Accu": history["test_acu"][-1],

                # average of per class --> P = precision = TP/(TP + FP) TP true positives FP false positives
                "Mean P": np.mean(precisions),
                "Mean R": np.mean(recalls),  # Just averaged the CM diagonal, R = recall = elem of diag

                "Macro F1": np.mean(f1_scores),  # average per class of 2(P*R)/(P+R)

                "Worst Class": np.min(recalls),

                "TV gap": generalization_gap,
                # "Loss Gap": loss_gap,

                "CM Deviation": cm_deviation,

                "Training Time": history["training_time"],
                "Max VRAM": history["GPU_max_VRAM"]
            }

        df = pd.DataFrame(data).T

        print(df.to_string(
            float_format=lambda x: f"{x:.2f}",
        ))

        # Best experiments according to individual metrics
        best_test = max(data, key=lambda e: data[e]["Test Accu"])
        best_valid = max(data, key=lambda e: data[e]["Valid Accu"])
        best_f1 = max(data, key=lambda e: data[e]["Macro F1"])
        best_worst_recall = max(data, key=lambda e: data[e]["Worst Class"])
        best_cm = min(data, key=lambda e: data[e]["CM Deviation"])
        best_training_time = min(data, key=lambda e: data[e]["Training Time"])
        best_VRAM = min(data, key=lambda e: data[e]["Max VRAM"])

        min_train_valid_gap = min(data, key=lambda e: abs(data[e]["TV gap"]))
        max_train_valid_gap = max(data, key=lambda e: abs(data[e]["TV gap"]))

        print("\n--- Best experiments ---")
        print(f"Max Test Accu      -> {best_test}: {data[best_test]['Test Accu']:.4f}")
        print(f"Max Valid Accu     -> {best_valid}: {data[best_valid]['Valid Accu']:.4f}")
        print(f"Max Macro F1           -> {best_f1}: {data[best_f1]['Macro F1']:.4f}")
        print(f"Best Worst-Class Recall -> {best_worst_recall}: {data[best_worst_recall]['Worst Class']:.4f}")
        print(f"Min CM Deviation       -> {best_cm}: {data[best_cm]['CM Deviation']:.4f}")
        print(f"Min Training Time       -> {best_training_time}: {data[best_cm]['Training Time']:.1f}s")
        print(f"Min VRAM       -> {best_VRAM}: {data[best_cm]['Max VRAM']} MB")
        print(f"Min abs Generalization Gap  -> {min_train_valid_gap}: {data[min_train_valid_gap]['TV gap']}")
        print(f"Max abs Generalization Gap  -> {max_train_valid_gap}: {data[max_train_valid_gap]['TV gap']}")


def show_confusion_matrix(confusion_matrix, title):
    cm = confusion_matrix.cpu()
    cm = cm.float() / cm.sum(dim=1, keepdim=True)  # normalized
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(FashionMNISTData.CLASS_NAMES.values()),
        yticklabels=list(FashionMNISTData.CLASS_NAMES.values())
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix {title}")

    plt.tight_layout()
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
