"""Дообучение ImageNet-моделей с кросс-валидацией по патчам."""

import argparse
import csv
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
}
METRIC_NAMES = ("loss", "accuracy", "macro_f1", "balanced_accuracy", "quadratic_kappa",
                "roc_auc_ovr_macro")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.reference_loss = float("inf")
        self.bad_epochs = 0

    def update(self, loss):
        if not np.isfinite(loss):
            raise ValueError(f"Validation loss должен быть конечным числом, получено {loss}")
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss
        # min_delta влияет на остановку; checkpoint всегда выбирается по минимуму loss.
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return is_best, self.patience > 0 and self.bad_epochs >= self.patience


def set_seed(seed, deterministic=False):
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def find_weights_file(weights, models_dir, cache_dir):
    """Ищем файл выбранных весов в общей папке, затем в локальном кеше."""
    if weights is None:
        return None
    filename = Path(urlparse(weights.url).path).name
    direct = models_dir / filename
    if direct.is_file():
        return direct
    if models_dir.is_dir():
        for path in models_dir.rglob(filename):
            if path.is_file():
                return path
    cached = cache_dir / filename
    return cached if cached.is_file() else None


def create_model(name, weights, num_classes, weights_file=None):
    model = MODELS[name][0](weights=None if weights_file is not None else weights)
    if weights_file is not None:
        # Загружаем исходную ImageNet-модель до замены её классификатора.
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    if name == "efficientnet_b0":
        # Сохраняем штатный Dropout перед последним линейным слоем.
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def classification_metrics(targets, probabilities, classes):
    labels = np.arange(len(classes))
    predictions = probabilities.argmax(axis=1)
    report = classification_report(targets, predictions, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    # One-vs-rest работает и для двух классов; без положительных/отрицательных AUC не определён.
    aucs = [roc_auc_score(targets == label, probabilities[:, label])
            if 0 < np.count_nonzero(targets == label) < len(targets) else float("nan")
            for label in labels]
    kappa = (cohen_kappa_score(targets, predictions, labels=labels, weights="quadratic")
             if np.unique(np.concatenate((targets, predictions))).size > 1 else float("nan"))
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": report["macro avg"]["f1-score"],
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "quadratic_kappa": kappa,
        "roc_auc_ovr_macro": float(np.mean(aucs)),
    }
    return metrics, report, aucs


def save_evaluation(output, prefix, targets, probabilities, classes):
    _, report, aucs = classification_metrics(targets, probabilities, classes)
    with (output / f"{prefix}_classification_report.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1", "support", "roc_auc_ovr"])
        for name, auc in zip(classes, aucs):
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"],
                             row["support"], auc])

    counts = confusion_matrix(targets, probabilities.argmax(axis=1), labels=range(len(classes)))
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    for suffix, values in (("confusion_matrix", counts),
                           ("confusion_matrix_normalized", normalized)):
        with (output / f"{prefix}_{suffix}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true / predicted", *classes])
            writer.writerows([name, *row] for name, row in zip(classes, values))

        is_normalized = suffix.endswith("normalized")
        displayed = values * 100 if is_normalized else values
        size = max(6, len(classes) * 1.1)
        figure = Figure(figsize=(size + 1, size), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        heatmap = ax.imshow(displayed, cmap="Blues", vmin=0,
                            vmax=100 if is_normalized else None)
        figure.colorbar(heatmap, ax=ax, label="% of true class" if is_normalized else "Patches")
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes, xlabel="Predicted class",
               ylabel="True class", title=f"{prefix}: confusion matrix")
        ax.tick_params(axis="x", labelrotation=30)
        for i, row in enumerate(displayed):
            for j, value in enumerate(row):
                label = f"{value:.1f}%" if is_normalized else str(int(value))
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if heatmap.norm(value) > 0.5 else "black")
        figure.savefig(output / f"{prefix}_{suffix}.png", dpi=150)
        figure.clear()


def save_predictions(output, dataset, folds, probabilities):
    predictions = probabilities.argmax(axis=1)
    with (output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "path", "fold", "true_class", "predicted_class",
                         *[f"prob_{name}" for name in dataset.classes]])
        for i, (path, target) in enumerate(dataset.samples):
            writer.writerow([i, Path(path).relative_to(dataset.root).as_posix(), folds[i],
                             dataset.classes[target], dataset.classes[predictions[i]],
                             *probabilities[i]])


def run_epoch(model, loader, criterion, device, optimizer=None, use_bf16=False):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    targets, probabilities = [], []
    memory_format = torch.channels_last if device.type == "cuda" else torch.contiguous_format

    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=memory_format)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            # BF16 не требует GradScaler; параметры модели остаются в FP32.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        # Переносим метрики на CPU один раз за эпоху, без синхронизации каждого батча.
        total_loss += loss.detach() * labels.size(0)
        if not training:
            targets.append(labels.detach())
            probabilities.append(logits.detach().float().softmax(dim=1))

    loss = total_loss.item() / len(loader.dataset)
    if training:
        return loss, None, None
    return loss, torch.cat(targets).cpu().numpy(), torch.cat(probabilities).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("Patch-based-dataset/train_data_patch"),
                        help="Папка с подпапками классов; только обучающая часть")
    parser.add_argument("--model", choices=MODELS, default="resnet18")
    parser.add_argument("--weights", default="IMAGENET1K_V1",
                        help="IMAGENET1K_V1, IMAGENET1K_V2 (ResNet50), DEFAULT или none")
    parser.add_argument("--models-dir", type=Path, default=Path("/raid/models"),
                        help="Общая папка ImageNet-весов; поиск также во вложенных папках")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50, help="Максимум эпох на фолд; по умолчанию 50")
    parser.add_argument("--patience", type=int, default=5,
                        help="Эпох без улучшения validation loss до остановки; 0 — отключить")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Минимальное снижение validation loss для сброса patience")
    parser.add_argument("--batch-size", type=int, help="По умолчанию 256 на CUDA, 16 на CPU")
    parser.add_argument("--workers", type=int, help="Загрузка данных: до 8 на CUDA, 0 на CPU")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--deterministic", action="store_true",
                        help="Строгая воспроизводимость; может снизить скорость")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна. На сервере нужен PyTorch с поддержкой CUDA")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if args.batch_size is None:
        args.batch_size = 256 if use_cuda else 16
    if args.workers is None:
        args.workers = min(8, os.cpu_count() or 1) if use_cuda else 0
    if args.workers < 0:
        parser.error("workers должен быть >= 0")
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Нужны folds >= 2, epochs >= 1, batch-size >= 1 и lr > 0")
    if args.patience < 0 or not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("Нужны patience >= 0 и конечное min-delta >= 0")
    if not 0 <= args.seed < 2**32:
        parser.error("seed должен быть от 0 до 2**32 - 1")
    if not args.data.is_dir():
        parser.error(f"Нет папки {args.data}. Распакуйте train_data_patch.zip (см. README.md)")
    use_bf16 = use_cuda and args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        parser.error("Эта GPU не поддерживает BF16. Укажите --precision fp32")
    torch.set_float32_matmul_precision("high" if use_cuda and not args.deterministic else "highest")

    weights_enum = MODELS[args.model][1]
    try:
        weights = None if args.weights.lower() == "none" else weights_enum[args.weights]
    except KeyError:
        parser.error(f"Для {args.model} доступны: {', '.join(weights_enum.__members__)}, none")

    # Предобработка соответствует выбранным ImageNet-весам.
    preprocess = (weights or weights_enum.IMAGENET1K_V1).transforms()
    train_transform = transforms.Compose([
        preprocess,
        # Отражаем уже уменьшенные патчи 224x224, а не исходные 1024x1024.
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    train_data = datasets.ImageFolder(args.data, transform=train_transform)
    val_data = datasets.ImageFolder(args.data, transform=preprocess)
    targets = np.array(train_data.targets)
    if len(train_data.classes) < 2 or np.bincount(targets).min() < args.folds:
        parser.error("Нужны хотя бы 2 класса и не меньше folds изображений в каждом классе")

    cache_dir = args.output / "_cache"
    torch.hub.set_dir(str(cache_dir))
    weights_file = find_weights_file(weights, args.models_dir, cache_dir / "checkpoints")
    output = args.output / f"{args.model}_{args.weights}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name() if use_cuda else "CPU"
    print(f"Device: {device_name} | Classes: {train_data.class_to_idx}", flush=True)
    print(f"Precision: {'bf16' if use_bf16 else 'fp32'} | Batch: {args.batch_size} | "
          f"Workers: {args.workers} | Deterministic: {args.deterministic}", flush=True)
    weights_source = (str(weights_file) if weights_file is not None else
                      f"torchvision -> {cache_dir / 'checkpoints'}" if weights is not None else
                      "random initialization")
    print(f"Weights: {weights.name if weights is not None else 'none'} | "
          f"Source: {weights_source}", flush=True)
    print(f"Epoch limit: {args.epochs} | Early stopping: val_loss, "
          f"patience={args.patience} (0 = disabled), min_delta={args.min_delta}", flush=True)
    print(f"Images: {len(train_data)} | Results: {output}", flush=True)

    # Один seed даёт одинаковые разбиения для сравнения моделей.
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = []
    fold_sizes = []
    oof_probabilities = np.empty((len(train_data), len(train_data.classes)), dtype=np.float32)
    oof_folds = np.zeros(len(train_data), dtype=int)
    with ((output / "metrics.csv").open("w", newline="", encoding="utf-8") as file,
          (output / "history.csv").open("w", newline="", encoding="utf-8") as history_file):
        writer = csv.DictWriter(file, fieldnames=["fold", *METRIC_NAMES, "best_epoch", "epochs_run"])
        writer.writeheader()
        history = csv.DictWriter(history_file, fieldnames=[
            "fold", "epoch", "train_loss", *[f"val_{name}" for name in METRIC_NAMES],
            "is_best", "time_seconds"])
        history.writeheader()
        for fold, (train_idx, val_idx) in enumerate(cv.split(targets, targets), start=1):
            fold_seed = (args.seed + fold - 1) % 2**32
            set_seed(fold_seed, args.deterministic)
            # Отдельные Dataset сохраняют аугментации только в train.
            loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                                  pin_memory=use_cuda, persistent_workers=args.workers > 0,
                                  worker_init_fn=seed_worker)
            train_loader = DataLoader(Subset(train_data, train_idx),
                                      shuffle=True, **loader_options,
                                      generator=torch.Generator().manual_seed(fold_seed))
            val_loader = DataLoader(Subset(val_data, val_idx),
                                    shuffle=False, **loader_options,
                                    generator=torch.Generator().manual_seed(fold_seed))
            # Каждый фолд начинает с исходных весов, а не с предыдущего фолда.
            memory_format = torch.channels_last if use_cuda else torch.contiguous_format
            model = create_model(args.model, weights, len(train_data.classes), weights_file).to(
                device=device, memory_format=memory_format)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=use_cuda)
            criterion = nn.CrossEntropyLoss()
            early_stopping = EarlyStopping(args.patience, args.min_delta)
            best_epoch, best_metrics, best_probabilities = 0, None, None

            for epoch in range(1, args.epochs + 1):
                started = time.perf_counter()
                train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, use_bf16)
                val_loss, val_targets, probabilities = run_epoch(
                    model, val_loader, criterion, device, use_bf16=use_bf16)
                metrics, _, _ = classification_metrics(val_targets, probabilities, train_data.classes)
                metrics = {"loss": val_loss, **metrics}
                is_best, should_stop = early_stopping.update(val_loss)
                if is_best:
                    best_epoch = epoch
                    best_metrics = {name: float(value) for name, value in metrics.items()}
                    best_probabilities = probabilities.copy()
                    # Сохраняем сразу: файл остаётся доступен и при прерывании обучения.
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model": args.model,
                        "weights": args.weights,
                        "classes": train_data.classes,
                        "seed": args.seed,
                        "fold": fold,
                        "epochs": args.epochs,
                        "best_epoch": best_epoch,
                        "best_metrics": best_metrics,
                        "patience": args.patience,
                        "min_delta": args.min_delta,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "folds": args.folds,
                        "workers": args.workers,
                        "precision": "bf16" if use_bf16 else "fp32",
                        "deterministic": args.deterministic,
                    }, output / f"fold_{fold}.pt")
                elapsed = time.perf_counter() - started
                history.writerow({"fold": fold, "epoch": epoch, "train_loss": train_loss,
                                  **{f"val_{name}": value for name, value in metrics.items()},
                                  "is_best": is_best, "time_seconds": elapsed})
                history_file.flush()
                print(f"Fold {fold}/{args.folds} | Epoch {epoch}/{args.epochs} | "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                      + " ".join(f"{name}={metrics[name]:.4f}" for name in METRIC_NAMES[1:])
                      + f" | best_epoch={best_epoch} time={elapsed:.1f}s", flush=True)
                if should_stop:
                    print(f"Fold {fold}: early stopping at epoch {epoch}; "
                          f"best epoch={best_epoch}, val_loss={best_metrics['loss']:.4f}", flush=True)
                    break

            # Итог, OOF-предсказания и checkpoint относятся к одной лучшей эпохе.
            scores.append([best_metrics[name] for name in METRIC_NAMES])
            fold_sizes.append(len(val_idx))
            writer.writerow({"fold": fold, **best_metrics, "best_epoch": best_epoch, "epochs_run": epoch})
            file.flush()
            # val_loader не перемешивается: порядок предсказаний совпадает с val_idx.
            oof_probabilities[val_idx] = best_probabilities
            oof_folds[val_idx] = fold
            save_evaluation(output, f"fold_{fold}", targets[val_idx], best_probabilities, train_data.classes)
            del model, optimizer, train_loader, val_loader

        means, stds = np.mean(scores, axis=0), np.std(scores, axis=0)
        writer.writerow({"fold": "mean", **dict(zip(METRIC_NAMES, means))})
        writer.writerow({"fold": "std", **dict(zip(METRIC_NAMES, stds))})
        oof_metrics, _, _ = classification_metrics(targets, oof_probabilities, train_data.classes)
        oof_metrics = {"loss": np.average(np.array(scores)[:, 0], weights=fold_sizes), **oof_metrics}
        writer.writerow({"fold": "oof", **oof_metrics})

    save_evaluation(output, "oof", targets, oof_probabilities, train_data.classes)
    save_predictions(output, val_data, oof_folds, oof_probabilities)
    for name, mean, std in zip(METRIC_NAMES, means, stds):
        print(f"CV {name}: {mean:.4f} +/- {std:.4f}")
    print("OOF: " + " ".join(f"{name}={value:.4f}" for name, value in oof_metrics.items()))


if __name__ == "__main__":
    main()
