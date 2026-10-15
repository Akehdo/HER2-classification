import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import main


class WeightsTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(main.__file__).resolve().parent / "runs"
        test_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="_weights_test_", dir=test_root)
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(test_root.resolve()))
        self.addCleanup(temporary.cleanup)
        self.shared = self.root / "shared"
        self.cache = self.root / "cache" / "checkpoints"
        self.shared.mkdir()
        self.cache.mkdir(parents=True)

    def test_matches_model_and_weights_version_in_nested_shared_folder(self):
        filenames = ["resnet18-f37072fd.pth", "resnet50-0676ba61.pth", "resnet50-11ad3fa6.pth",
                     "efficientnet_b0_rwightman-7f5810bc.pth"]
        nested = self.shared / "torchvision" / "checkpoints"
        nested.mkdir(parents=True)
        for filename in filenames:
            (nested / filename).touch()
        cases = [
            ("resnet18", "IMAGENET1K_V1", filenames[0]),
            ("resnet18", "DEFAULT", filenames[0]),
            ("resnet50", "IMAGENET1K_V1", filenames[1]),
            ("resnet50", "IMAGENET1K_V2", filenames[2]),
            ("resnet50", "DEFAULT", filenames[2]),
            ("efficientnet_b0", "IMAGENET1K_V1", filenames[3]),
            ("efficientnet_b0", "DEFAULT", filenames[3]),
        ]
        for model, version, filename in cases:
            with self.subTest(model=model, version=version):
                weights = main.MODELS[model][1][version]
                self.assertEqual(main.find_weights_file(weights, self.shared, self.cache), nested / filename)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertEqual(sorted(path.name for path in nested.iterdir()), sorted(filenames))

    def test_prefers_shared_file_over_local_cache(self):
        filename = "resnet18-f37072fd.pth"
        (self.shared / filename).touch()
        (self.cache / filename).touch()
        self.assertEqual(main.find_weights_file(main.models.ResNet18_Weights.DEFAULT,
                                               self.shared, self.cache), self.shared / filename)

    def test_uses_local_cache_without_matching_shared_file(self):
        (self.shared / "resnet50-0676ba61.pth").touch()
        cached = self.cache / "resnet18-f37072fd.pth"
        cached.touch()
        weights = main.models.ResNet18_Weights.DEFAULT
        self.assertEqual(main.find_weights_file(weights, self.shared, self.cache), cached)
        self.assertEqual(main.find_weights_file(weights, self.root / "absent", self.cache), cached)

    def test_missing_weights_and_random_initialization(self):
        self.assertIsNone(main.find_weights_file(main.models.ResNet18_Weights.DEFAULT,
                                                self.shared, self.cache))
        with mock.patch.object(Path, "is_file", side_effect=AssertionError("Unexpected search")):
            self.assertIsNone(main.find_weights_file(None, self.shared, self.cache))

    def test_loads_original_checkpoint_before_replacing_classifier(self):
        class TinyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 2)
                self.fc = nn.Linear(2, 1000)

        original = TinyResNet()
        with torch.no_grad():
            original.backbone.weight.fill_(0.25)
        checkpoint = self.shared / "resnet18-f37072fd.pth"
        torch.save(original.state_dict(), checkpoint)
        factory = mock.Mock(side_effect=lambda **kwargs: TinyResNet())
        weights = main.models.ResNet18_Weights.DEFAULT
        with mock.patch.dict(main.MODELS, {"resnet18": (factory, main.models.ResNet18_Weights)}):
            # Несколько фолдов должны начинать с одних и тех же исходных весов.
            for _ in range(2):
                model = main.create_model("resnet18", weights, 4, checkpoint)
                torch.testing.assert_close(model.backbone.weight, original.backbone.weight)
                self.assertEqual(model.fc.out_features, 4)
                with torch.no_grad():
                    model.backbone.weight.zero_()
        self.assertEqual(factory.call_args_list, [mock.call(weights=None), mock.call(weights=None)])
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_missing_file_preserves_torchvision_download_path(self):
        factory = mock.Mock(return_value=nn.Module())
        factory.return_value.fc = nn.Linear(2, 1000)
        weights = main.models.ResNet50_Weights.IMAGENET1K_V1
        with mock.patch.dict(main.MODELS, {"resnet50": (factory, main.models.ResNet50_Weights)}):
            model = main.create_model("resnet50", weights, 4)
        factory.assert_called_once_with(weights=weights)
        self.assertEqual(model.fc.out_features, 4)


if __name__ == "__main__":
    unittest.main()
