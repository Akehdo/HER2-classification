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

import main


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_including_equal_loss(self):
        stopping = main.EarlyStopping(patience=2)
        self.assertEqual(stopping.update(1.0), (True, False))
        self.assertEqual(stopping.update(0.5), (True, False))
        self.assertEqual(stopping.update(0.5), (False, False))
        self.assertEqual(stopping.update(0.6), (False, True))

    def test_improvement_resets_counter(self):
        stopping = main.EarlyStopping(patience=2)
        for loss in [1.0, 1.1, 0.9, 1.0]:
            self.assertFalse(stopping.update(loss)[1])
        self.assertEqual(stopping.bad_epochs, 1)
        self.assertTrue(stopping.update(1.0)[1])

    def test_min_delta_does_not_discard_actual_best_checkpoint(self):
        stopping = main.EarlyStopping(patience=3, min_delta=0.25)
        stopping.update(1.0)
        # Снижение ровно на min_delta ещё не сбрасывает счётчик.
        self.assertEqual(stopping.update(0.75), (True, False))
        self.assertEqual(stopping.bad_epochs, 1)
        self.assertEqual(stopping.best_loss, 0.75)
        self.assertEqual(stopping.reference_loss, 1.0)
        stopping.update(0.5)
        self.assertEqual(stopping.bad_epochs, 0)

    def test_patience_zero_disables_stopping(self):
        stopping = main.EarlyStopping(patience=0)
        for loss in [1.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
            self.assertFalse(stopping.update(loss)[1])
        self.assertEqual(stopping.best_loss, 0.5)

    def test_nonfinite_loss_is_rejected(self):
        for loss in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                main.EarlyStopping().update(loss)

    def test_invalid_cli_settings(self):
        for arguments in [["--patience", "-1"], ["--min-delta", "-0.1"],
                          ["--min-delta", "nan"], ["--min-delta", "inf"]]:
            with (self.subTest(arguments=arguments),
                  mock.patch.object(sys, "argv", ["main.py", *arguments]),
                  contextlib.redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit) as raised):
                main.main()
            self.assertEqual(raised.exception.code, 2)

    def test_best_weights_metrics_and_oof_survive_later_worse_epochs(self):
        classes = ["class_0", "class_1+", "class_2+", "class_3+"]
        test_root = Path(main.__file__).resolve().parent / "runs"
        test_root.mkdir(exist_ok=True)
        cases = [([], 7, 50, 5), (["--epochs", "4", "--patience", "0"], 4, 4, 0),
                 (["--epochs", "3"], 3, 3, 5)]
        for arguments, expected_epochs, limit, patience in cases:
            with (self.subTest(arguments=arguments),
                  tempfile.TemporaryDirectory(prefix="_early_stopping_test_", dir=test_root) as temp):
                root = Path(temp).resolve()
                self.assertTrue(root.is_relative_to(test_root.resolve()))
                data = root / "data"
                for name in classes:
                    folder = data / name
                    folder.mkdir(parents=True)
                    for sample in range(2):
                        Image.new("RGB", (8, 8)).save(folder / f"{sample}.png")

                def tiny_model(*args):
                    model = nn.Module()
                    model.fc = nn.Linear(3, 4)
                    model.test_epoch = 0
                    return model

                def fake_epoch(model, loader, criterion, device, optimizer=None, use_bf16=False):
                    if optimizer is not None:
                        model.test_epoch += 1
                        with torch.no_grad():
                            model.fc.weight.fill_(model.test_epoch)
                        return 0.1, None, None
                    labels = np.array([loader.dataset.dataset.targets[i] for i in loader.dataset.indices])
                    probabilities = np.eye(4)[labels] * 0.8 + 0.05
                    if model.test_epoch != 2:
                        probabilities = np.roll(probabilities, 1, axis=1)
                    loss = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][model.test_epoch - 1]
                    return loss, labels, probabilities

                argv = ["main.py", "--data", str(data), "--output", str(root / "runs"),
                        "--weights", "none", "--device", "cpu", "--folds", "2", "--workers", "0",
                        *arguments]
                with (mock.patch.object(sys, "argv", argv),
                      mock.patch.object(main, "create_model", side_effect=tiny_model),
                      mock.patch.object(main, "run_epoch", side_effect=fake_epoch) as epochs,
                      mock.patch.object(main, "save_evaluation") as evaluations,
                      contextlib.redirect_stdout(io.StringIO())):
                    main.main()
                self.assertEqual(epochs.call_count, expected_epochs * 2 * 2)
                output = root / "runs" / "resnet18_none_seed42"

                def rows(filename):
                    with (output / filename).open(newline="", encoding="utf-8") as file:
                        return list(csv.DictReader(file))

                history, summary = rows("history.csv"), rows("metrics.csv")
                self.assertEqual(len(history), expected_epochs * 2)
                self.assertEqual(sum(row["is_best"] == "True" for row in history), 4)
                for fold in [1, 2]:
                    checkpoint = torch.load(output / f"fold_{fold}.pt", weights_only=True)
                    torch.testing.assert_close(checkpoint["model_state_dict"]["fc.weight"],
                                               torch.full((4, 3), 2.0))
                    self.assertEqual(checkpoint["best_epoch"], 2)
                    self.assertEqual(checkpoint["epochs"], limit)
                    self.assertEqual(checkpoint["patience"], patience)
                    self.assertEqual(checkpoint["best_metrics"]["loss"], 0.5)
                    self.assertEqual(summary[fold - 1]["best_epoch"], "2")
                    self.assertEqual(summary[fold - 1]["epochs_run"], str(expected_epochs))
                for row in summary:
                    if row["fold"] != "std":
                        self.assertEqual(float(row["accuracy"]), 1)
                        self.assertEqual(float(row["loss"]), 0.5)
                self.assertTrue(all(row["true_class"] == row["predicted_class"]
                                    for row in rows("oof_predictions.csv")))
                self.assertEqual(evaluations.call_count, 3)
                for call in evaluations.call_args_list:
                    _, _, labels, probabilities, _ = call.args
                    np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)


if __name__ == "__main__":
    unittest.main()
