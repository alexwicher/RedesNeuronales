import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from pract11.Guia11Reformat.src.ClassifierMLP import ClassifierMLP
from pract11.Guia11Reformat.src.Optimizer import create_optimizer
from pract11.Guia11Reformat.src.Trainer import Trainer
from pract11.Guia11Reformat.src.Visualizer import Visualizer


def runExperiment(config, data, seedOverride=None):
    # Configuración
    if seedOverride is not None: config.SEED = seedOverride
    device = config.setup()

    if config.show_sample:
        data.inspect_example()

    # Modelo
    torch.manual_seed(config.SEED)

    model = ClassifierMLP(
        p_dropout=config.DROPOUT_P, neuron_1=config.neuron_1, neuron_2=config.neuron_2,
    ).to(device)

    # Comprobar dimensiones
    if config.show_dimentions:
        images, labels = next(iter(data.train_loader))

        with torch.no_grad():
            classifications = model(images.to(device))

        print(f"images.shape: {images.shape}")
        print(f"labels.shape: {labels.shape}")

        n_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        print(f"Parámetros entrenables: {n_params:,}")

        print("\nDimensiones:")
        print("Entrada:", images.shape)
        print("Logits:", classifications.shape)

    # Función de pérdida y optimizador
    loss_fn = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        optimizer_name=config.OPTIMIZER_NAME,
        adam_lr=config.ADAM_LR,
        sgd_lr=config.SGD_LR
    )

    print("\nOptimizador:", config.OPTIMIZER_NAME)
    print("Dropout:", config.DROPOUT_P)

    # Untrained
    if config.show_untrained:
        Visualizer.showPredictions(model, device, data)

    # Entrenamiento
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device,
        config
    )
    start = time.perf_counter()
    trainer.fit(
        data.train_loader,
        data.valid_loader,
        epochs=config.EPOCHS
    )

    print(f"Training time: {time.perf_counter() - start:.2f} seconds")
    # Gráfico de pérdidas
    if config.show_loss: Visualizer.plot_history(trainer.history)

    # Evaluación final
    _, _, confusion_matrix = trainer.test(data.test_loader)

    # Trained
    if config.show_trained:
        Visualizer.showPredictions(model, device, data)
    return trainer, data, confusion_matrix


def pre_run(expies, data):
    results_together = {}
    for title, exp in expies.items():
        results = {}
        data.reset_generator()  # Reset before each experiment
        trainer, data, confusion_matrix = runExperiment(exp, data)
        results[title] = {"trainer": trainer, "data": data, "confusion_matrix": confusion_matrix}
        results_together[title] = {"trainer": trainer, "data": data, "confusion_matrix": confusion_matrix}
        Visualizer.showExpieResults(results)
    print("\n-----------------------------------Results TOGETHER:-----------------------------------\n")
    Visualizer.showExpieResults(results_together, show_confusion=False)
    Visualizer.showComparativeTable(results_together)
    return results_together


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cm_metrics(cm):
    # Convert tensor to numpy if necessary
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().cpu().numpy()

    cm = np.asarray(cm, dtype=float)

    # Normalize rows if matrix contains counts or percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(
        cm, row_sums,
        out=np.zeros_like(cm),
        where=row_sums != 0
    )

    recalls = np.diag(cm)

    column_sums = cm.sum(axis=0)

    precisions = np.divide(
        np.diag(cm),
        column_sums,
        out=np.zeros_like(column_sums),
        where=column_sums != 0
    )

    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(recalls),
        where=(precisions + recalls) != 0
    )

    return {
        "Macro Precision": np.mean(precisions),
        "Macro Recall": np.mean(recalls),
        "Macro F1": np.mean(f1),
        "Worst Class Recall": np.min(recalls),
        "CM Deviation": np.sum((cm - np.eye(cm.shape[0])) ** 2)
    }, cm


def seededPreRun(expies, data, seeds=(42, 123, 456, 789, 999)):
    results_together = {}

    for title, exp in expies.items():

        print(f"\nRunning experiment: {title}")

        runs = []
        confusion_matrices = []
        run_metrics = []

        for seed in seeds:
            print(f"  Seed: {seed}")

            trainer, run_data, cm = runExperiment(exp, data, seed)

            metrics, normalized_cm = get_cm_metrics(cm)

            history = trainer.history

            metrics.update({
                "Train Loss": history["train_loss"][-1],
                "Valid Loss": history["valid_loss"][-1],
                "Train Accuracy": history["train_acu"][-1],
                "Valid Accuracy": history["valid_acu"][-1],
                "Test Accuracy": history["test_acu"][-1],
                "Generalization Gap": (
                        history["train_acu"][-1] -
                        history["valid_acu"][-1]
                ),
                "Loss Gap": (
                        history["valid_loss"][-1] -
                        history["train_loss"][-1]
                )
            })

            runs.append({
                "seed": seed,
                "trainer": trainer,
                "data": run_data,
                "confusion_matrix": normalized_cm,
                "metrics": metrics
            })

            confusion_matrices.append(normalized_cm)
            run_metrics.append(metrics)

        # Average confusion matrix across runs
        mean_cm = np.mean(confusion_matrices, axis=0)

        # Average and standard deviation of each metric
        metrics_df = pd.DataFrame(run_metrics)

        results_together[title] = {
            "runs": runs,
            "confusion_matrix": mean_cm,
            "metrics_mean": metrics_df.mean().to_dict(),
            "metrics_std": metrics_df.std(ddof=1).to_dict(),
            "metrics_per_run": metrics_df
        }

    return results_together
