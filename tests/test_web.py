"""Проверка реального forward и HTTP-контракта без скачивания весов."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from fastapi.testclient import TestClient
from PIL import Image

import app
from inference import InvalidImage, Predictor, decode_image
from model_registry import MODELS, create_model


CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"]


def image_bytes(mode="RGB", format="PNG"):
    file = io.BytesIO()
    Image.new(mode, (320, 256), 130).save(file, format=format)
    return file.getvalue()


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        (app.ROOT / "runs").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="_web_test_", dir=app.ROOT / "runs")
        cls.root = Path(cls.temporary.name)
        cls.checkpoint = cls.root / "fold_1.pt"
        cls.original = create_model("resnet18", None, len(CLASSES)).eval()
        torch.save({"model": "resnet18", "weights": "IMAGENET1K_V1", "classes": CLASSES,
                    "model_state_dict": cls.original.state_dict(), "fold": 1, "best_epoch": 2},
                   cls.checkpoint)
        cls.context = contextlib.ExitStack()
        cls.client = cls.context.enter_context(TestClient(app.create_app(cls.checkpoint, "cpu")))

    @classmethod
    def tearDownClass(cls):
        cls.context.close()
        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_real_inference_matches_training_preprocessing_and_forward(self):
        image = Image.new("RGB", (320, 256), (105, 60, 155))
        predictor = self.client.app.state.predictor
        expected_input = MODELS["resnet18"][1].IMAGENET1K_V1.transforms()(image).unsqueeze(0)
        with torch.inference_mode():
            expected = self.original(expected_input).softmax(dim=1)[0]
        results, elapsed = predictor.predict([image])
        torch.testing.assert_close(torch.tensor(list(results[0]["probabilities"].values())), expected)
        self.assertFalse(predictor.model.training)
        self.assertEqual(results[0]["predicted_class"], CLASSES[int(expected.argmax())])
        self.assertGreater(elapsed, 0)

    def test_all_architectures_load_offline_and_preserve_class_order(self):
        for name in ("resnet50", "efficientnet_b0"):
            with self.subTest(model=name):
                classes = list(reversed(CLASSES))
                model = create_model(name, None, len(classes))
                path = self.root / f"{name}.pt"
                torch.save({"model": name, "weights": "DEFAULT", "classes": classes,
                            "model_state_dict": model.state_dict()}, path)
                with mock.patch.object(MODELS[name][1], "get_state_dict",
                                       side_effect=AssertionError("Unexpected download")):
                    predictor = Predictor(path, "cpu")
                results, _ = predictor.predict([Image.new("RGB", (256, 256))])
                self.assertEqual(list(results[0]["probabilities"]), classes)
                self.assertAlmostEqual(sum(results[0]["probabilities"].values()), 1, places=5)
                self.assertEqual(predictor.preprocess.crop_size, [224])
                del predictor, model

    def test_batch_http_contract_and_metadata(self):
        response = self.client.post("/predict", files=[
            ("files", ("first.png", image_bytes(), "image/png")),
            ("files", ("second.png", image_bytes("L"), "image/png")),
        ])
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual([row["filename"] for row in data["results"]], ["first.png", "second.png"])
        self.assertEqual(data["model"]["classes"], CLASSES)
        self.assertEqual(data["model"]["checkpoint"], "fold_1.pt")
        self.assertEqual(len(data["model"]["version"]), 12)
        self.assertEqual(data["model"]["best_epoch"], 2)
        for result in data["results"]:
            self.assertEqual((result["width"], result["height"]), (320, 256))
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=5)
            self.assertAlmostEqual(result["probability"], result["probabilities"][result["predicted_class"]])

    def test_health_static_files_and_openapi(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ready")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HER2", response.text)
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        for path in ["/static/app.js", "/static/styles.css", "/static/favicon.svg", "/docs"]:
            self.assertEqual(self.client.get(path).status_code, 200)
        operation = self.client.get("/openapi.json").json()["paths"]["/predict"]["post"]
        schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["properties"]["files"]["maxItems"], 16)

    def test_rejects_corrupt_files_unsupported_formats_and_empty_upload(self):
        for content in [b"not a real PNG", b"", image_bytes(format="BMP")]:
            with self.subTest(content=content[:10]):
                response = self.client.post("/predict", files={"files": ("patch.png", content, "image/png")})
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.post("/predict", json={}).status_code, 415)
        self.assertEqual(self.client.post("/predict", files={"other": ("a.png", image_bytes())}).status_code, 422)

    def test_enforces_file_count_and_size_limits(self):
        files = [("files", (f"{i}.png", b"x")) for i in range(17)]
        self.assertEqual(self.client.post("/predict", files=files).status_code, 400)
        with mock.patch.object(app, "MAX_FILE_BYTES", 5):
            response = self.client.post("/predict", files={"files": ("big.png", image_bytes())})
        self.assertEqual(response.status_code, 413)
        response = self.client.post("/predict", content=b"", headers={"content-length": str(app.MAX_REQUEST_BYTES + 1)})
        self.assertEqual(response.status_code, 413)
        with mock.patch.object(app, "MAX_REQUEST_BYTES", 5):
            response = self.client.post("/predict", content=iter([b"1234", b"5678"]),
                                        headers={"content-type": "multipart/form-data; boundary=test"})
        self.assertEqual(response.status_code, 413, response.text)

    def test_missing_checkpoint_is_explicit_and_never_returns_predictions(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(Path, "is_file", return_value=False):
            application = app.create_app()
        with TestClient(application) as client:
            self.assertEqual(client.get("/health").json()["status"], "not_ready")
            self.assertIsNone(client.get("/health").json()["model"])
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.post("/predict", files={"files": ("a.png", image_bytes())}).status_code, 503)

    def test_bad_checkpoint_does_not_break_the_interface(self):
        bad_path = self.root / "invalid.pt"
        bad_path.write_bytes(b"invalid")
        with self.assertLogs("her2", level="ERROR"):
            with TestClient(app.create_app(bad_path, "cpu")) as client:
                health = client.get("/health").json()
                self.assertEqual(health["status"], "not_ready")
                self.assertIn("Не удалось загрузить", health["message"])
                self.assertEqual(client.get("/").status_code, 200)

    def test_image_decoding_applies_orientation_and_rejects_multi_page_tiff(self):
        self.assertEqual(decode_image(image_bytes("L")).mode, "RGB")
        file = io.BytesIO()
        image = Image.new("RGB", (40, 20))
        exif = image.getexif()
        exif[274] = 6
        image.save(file, format="JPEG", exif=exif)
        self.assertEqual(decode_image(file.getvalue()).size, (20, 40))
        file = io.BytesIO()
        image.save(file, format="TIFF", save_all=True, append_images=[image])
        with self.assertRaises(InvalidImage):
            decode_image(file.getvalue())
        with mock.patch("inference.MAX_IMAGE_PIXELS", 1):
            with self.assertRaises(InvalidImage):
                decode_image(image_bytes())


if __name__ == "__main__":
    unittest.main()
