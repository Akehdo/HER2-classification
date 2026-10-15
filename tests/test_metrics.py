import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import main


CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        targets = np.array([0, 1, 2, 3])
        metrics, report, aucs = main.classification_metrics(targets, np.eye(4), CLASSES)
        for value in metrics.values():
            self.assertAlmostEqual(value, 1)
        np.testing.assert_allclose(aucs, 1)
        self.assertEqual(report["class_2+"]["support"], 1)

    def test_imbalanced_constant_predictions(self):
        targets = np.array([0, 0, 0, 1, 2, 3])
        probabilities = np.tile([0.6, 0.2, 0.1, 0.1], (len(targets), 1))
        metrics, _, aucs = main.classification_metrics(targets, probabilities, CLASSES)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.25)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 6)
        self.assertAlmostEqual(metrics["quadratic_kappa"], 0)
        np.testing.assert_allclose(aucs, 0.5)

    def test_kappa_penalizes_distant_errors(self):
        targets = np.array([0, 1, 2, 3])
        near, _, _ = main.classification_metrics(targets, np.eye(4)[[1, 0, 3, 2]], CLASSES)
        far, _, _ = main.classification_metrics(targets, np.eye(4)[[3, 2, 1, 0]], CLASSES)
        self.assertEqual(near["accuracy"], far["accuracy"])
        self.assertGreater(near["quadratic_kappa"], far["quadratic_kappa"])

    def test_auc_uses_probabilities_and_supports_binary(self):
        # Одинаковые argmax, но разное ранжирование вероятностей положительного класса.
        targets = np.array([0, 0, 1, 1])
        good = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
        bad = good[::-1]
        good_metrics, _, _ = main.classification_metrics(targets, good, CLASSES[:2])
        bad_metrics, _, _ = main.classification_metrics(targets, bad, CLASSES[:2])
        self.assertEqual(good_metrics["accuracy"], bad_metrics["accuracy"])
        self.assertAlmostEqual(good_metrics["roc_auc_ovr_macro"], 1)
        self.assertAlmostEqual(bad_metrics["roc_auc_ovr_macro"], 0)

    def test_undefined_auc_and_kappa(self):
        metrics, report, aucs = main.classification_metrics(
            np.array([0, 0]), np.array([[1, 0], [1, 0]]), CLASSES[:2])
        self.assertTrue(np.isnan(aucs).all())
        self.assertTrue(np.isnan(metrics["roc_auc_ovr_macro"]))
        self.assertTrue(np.isnan(metrics["quadratic_kappa"]))
        self.assertEqual(report["class_1+"]["recall"], 0)

    def test_epoch_preserves_order_and_weights_short_batch(self):
        logits = torch.tensor([[3.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 0, 1.0]])
        targets = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2, shuffle=False)
        loss, actual_targets, probabilities = main.run_epoch(
            nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertAlmostEqual(loss, nn.functional.cross_entropy(logits, targets).item(), places=6)
        np.testing.assert_array_equal(actual_targets, targets.numpy())
        np.testing.assert_allclose(probabilities, logits.softmax(dim=1).numpy())

    def test_main_saves_fold_and_oof_reports(self):
        test_root = Path(main.__file__).resolve().parent / "runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="_metrics_test_", dir=test_root) as temp:
            root = Path(temp).resolve()
            self.assertTrue(root.is_relative_to(test_root.resolve()))
            data = root / "data"
            for class_index, name in enumerate(CLASSES):
                folder = data / name
                folder.mkdir(parents=True)
                for sample in range(5):
                    Image.new("RGB", (24, 24), (class_index * 60, sample * 20, 90)).save(
                        folder / f"patch_{sample}.png")

            argv = ["main.py", "--data", str(data), "--output", str(root / "runs"),
                    "--weights", "none", "--device", "cpu", "--folds", "3", "--epochs", "2",
                    "--batch-size", "4", "--workers", "0"]
            # Проверяем весь CV и запись файлов без дорогого обучения ResNet.
            def tiny_model(name, weights, num_classes, weights_file=None):
                return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))

            with (mock.patch.object(sys, "argv", argv),
                  mock.patch.object(main, "create_model", side_effect=tiny_model),
                  contextlib.redirect_stdout(io.StringIO())):
                main.main()

            output = root / "runs" / "resnet18_none_seed42"
            history = read_csv(output / "history.csv")
            summary = read_csv(output / "metrics.csv")
            predictions = read_csv(output / "oof_predictions.csv")
            self.assertEqual(len(history), 6)
            self.assertEqual([row["fold"] for row in summary], ["1", "2", "3", "mean", "std", "oof"])
            self.assertEqual(len(predictions), 20)
            self.assertEqual(len({row["path"] for row in predictions}), 20)

            probabilities = np.array([[float(row[f"prob_{name}"]) for name in CLASSES]
                                      for row in predictions])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            targets = np.array([CLASSES.index(row["true_class"]) for row in predictions])
            for row, predicted in zip(predictions, probabilities.argmax(axis=1)):
                self.assertEqual(row["predicted_class"], CLASSES[predicted])

            splits = list(main.StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
                targets, targets))
            for fold, (_, val_idx) in enumerate(splits, 1):
                self.assertTrue(all(predictions[i]["fold"] == str(fold) for i in val_idx))
                best_epoch = min((row for row in history if row["fold"] == str(fold)),
                                 key=lambda row: float(row["val_loss"]))
                self.assertEqual(int(summary[fold - 1]["best_epoch"]), int(best_epoch["epoch"]))
                self.assertEqual(int(summary[fold - 1]["epochs_run"]), 2)
                for name in main.METRIC_NAMES:
                    self.assertEqual(float(summary[fold - 1][name]), float(best_epoch[f"val_{name}"]))
                self.assertTrue((output / f"fold_{fold}.pt").is_file())

            for prefix in ["fold_1", "fold_2", "fold_3", "oof"]:
                rows = read_csv(output / f"{prefix}_confusion_matrix.csv")
                counts = np.array([[int(row[name]) for name in CLASSES] for row in rows])
                normalized_rows = read_csv(output / f"{prefix}_confusion_matrix_normalized.csv")
                normalized = np.array([[float(row[name]) for name in CLASSES] for row in normalized_rows])
                np.testing.assert_allclose(normalized.sum(axis=1), 1)
                np.testing.assert_allclose(normalized, counts / counts.sum(axis=1, keepdims=True))
                report = read_csv(output / f"{prefix}_classification_report.csv")
                self.assertEqual([row["class"] for row in report], CLASSES)
                for suffix in ["confusion_matrix", "confusion_matrix_normalized"]:
                    with Image.open(output / f"{prefix}_{suffix}.png") as image:
                        image.verify()
                if prefix == "oof":
                    expected = main.confusion_matrix(targets, probabilities.argmax(axis=1), labels=range(4))
                    np.testing.assert_array_equal(counts, expected)
                    self.assertEqual(counts.sum(), 20)

            expected_oof, _, _ = main.classification_metrics(targets, probabilities, CLASSES)
            for name, expected in expected_oof.items():
                self.assertAlmostEqual(float(summary[-1][name]), expected)
            fold_values = np.array([[float(row[name]) for name in main.METRIC_NAMES]
                                    for row in summary[:3]])
            np.testing.assert_allclose([float(summary[3][name]) for name in main.METRIC_NAMES],
                                       fold_values.mean(axis=0))
            np.testing.assert_allclose([float(summary[4][name]) for name in main.METRIC_NAMES],
                                       fold_values.std(axis=0))
            expected_loss = np.average(fold_values[:, 0], weights=[len(val) for _, val in splits])
            self.assertAlmostEqual(float(summary[-1]["loss"]), expected_loss)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
