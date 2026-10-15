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


class EfficientNetTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(main.__file__).resolve().parent / "runs"
        test_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="_efficientnet_test_", dir=test_root)
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(test_root.resolve()))
        self.addCleanup(temporary.cleanup)

    def test_four_class_head_preserves_dropout_and_backpropagates(self):
        model = main.create_model("efficientnet_b0", None, 4)
        self.assertIsInstance(model.classifier[0], nn.Dropout)
        self.assertEqual(model.classifier[0].p, 0.2)
        self.assertEqual(model.classifier[-1].out_features, 4)
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 4))
        nn.CrossEntropyLoss()(logits, torch.tensor([0, 3])).backward()
        for gradient in [model.features[0][0].weight.grad, model.classifier[-1].weight.grad]:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_loads_original_shared_weights_before_replacing_head(self):
        original = main.models.efficientnet_b0(weights=None)
        with torch.no_grad():
            original.features[0][0].weight.fill_(0.25)
        shared = self.root / "models"
        shared.mkdir()
        checkpoint = shared / "efficientnet_b0_rwightman-7f5810bc.pth"
        torch.save(original.state_dict(), checkpoint)
        weights = main.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        path = main.find_weights_file(weights, shared, self.root / "absent_cache")
        with mock.patch.object(main.models.EfficientNet_B0_Weights, "get_state_dict",
                               side_effect=AssertionError("Unexpected download")):
            model = main.create_model("efficientnet_b0", weights, 4, path)
        torch.testing.assert_close(model.features[0][0].weight, original.features[0][0].weight)
        self.assertEqual(model.classifier[-1].weight.shape, (4, 1280))
        self.assertIsInstance(model.classifier[0], nn.Dropout)
        self.assertEqual(list(shared.iterdir()), [checkpoint])
        self.assertFalse((self.root / "absent_cache").exists())

    def test_cli_trains_two_folds_and_saves_reports(self):
        classes = ["class_0", "class_1+", "class_2+", "class_3+"]
        data = self.root / "data"
        for class_index, name in enumerate(classes):
            folder = data / name
            folder.mkdir(parents=True)
            for sample in range(2):
                Image.new("RGB", (24, 24), (class_index * 60, sample * 80, 90)).save(
                    folder / f"{sample}.png")
        argv = ["main.py", "--model", "efficientnet_b0", "--weights", "none", "--data", str(data),
                "--output", str(self.root / "runs"), "--device", "cpu", "--folds", "2",
                "--epochs", "1", "--batch-size", "4", "--workers", "0", "--seed", "123"]
        weights = main.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        preprocessing = mock.Mock(return_value=preprocess)
        with (mock.patch.object(sys, "argv", argv),
              mock.patch.object(type(weights), "transforms", new_callable=mock.PropertyMock,
                                return_value=preprocessing),
              contextlib.redirect_stdout(io.StringIO())):
            main.main()
        preprocessing.assert_called_once_with()
        output = self.root / "runs" / "efficientnet_b0_none_seed123"

        def rows(filename):
            with (output / filename).open(newline="", encoding="utf-8") as file:
                return list(csv.DictReader(file))

        history, metrics = rows("history.csv"), rows("metrics.csv")
        self.assertEqual(len(history), 2)
        self.assertTrue(all(np.isfinite(float(row["val_loss"])) for row in history))
        self.assertEqual([row["fold"] for row in metrics], ["1", "2", "mean", "std", "oof"])
        for fold in [1, 2]:
            checkpoint = torch.load(output / f"fold_{fold}.pt", weights_only=True)
            self.assertEqual(checkpoint["model"], "efficientnet_b0")
            self.assertEqual(checkpoint["best_epoch"], 1)
            self.assertEqual(checkpoint["model_state_dict"]["classifier.1.weight"].shape, (4, 1280))
            self.assertNotIn("fc.weight", checkpoint["model_state_dict"])
        predictions = rows("oof_predictions.csv")
        self.assertEqual(len(predictions), 8)
        self.assertEqual(len({row["path"] for row in predictions}), 8)
        probabilities = [[float(row[f"prob_{name}"]) for name in classes] for row in predictions]
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1, atol=1e-6)
        for prefix in ["fold_1", "fold_2", "oof"]:
            self.assertEqual([row["class"] for row in rows(f"{prefix}_classification_report.csv")], classes)
            for suffix in ["confusion_matrix", "confusion_matrix_normalized"]:
                with Image.open(output / f"{prefix}_{suffix}.png") as image:
                    image.verify()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
