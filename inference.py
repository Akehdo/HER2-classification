"""Инференс из checkpoint, созданного main.py. Загрузка не обращается к сети."""

import hashlib
import io
import threading
import time
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from model_registry import MODELS, create_model


MAX_IMAGE_PIXELS = 16_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF", "WEBP"}


class InvalidImage(ValueError):
    pass


def decode_image(data):
    """Проверяем декодированное изображение, а не расширение пользовательского файла."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Поддерживаются PNG, JPEG, TIFF и WebP.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("Изображение слишком большое: максимум 16 мегапикселей.")
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("Загрузите один патч: многостраничные изображения не поддерживаются.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise InvalidImage("Не удалось прочитать изображение. Проверьте файл патча.") from error


class Predictor:
    def __init__(self, checkpoint_path, device="auto"):
        path = Path(checkpoint_path)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Устройство должно быть auto, cpu или cuda.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA недоступна. Выберите --device cpu или проверьте PyTorch/CUDA.")
        self.device = torch.device(
            "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Ожидается checkpoint из main.py, содержащий веса и метаданные.")
        self.name = checkpoint.get("model")
        if self.name not in MODELS:
            raise ValueError("Checkpoint содержит неизвестную архитектуру модели.")
        self.classes = checkpoint.get("classes")
        if (not isinstance(self.classes, list) or len(self.classes) < 2
                or not all(isinstance(name, str) and name for name in self.classes)
                or len(set(self.classes)) != len(self.classes)):
            raise ValueError("В checkpoint отсутствует корректный список classes.")
        weights_enum = MODELS[self.name][1]
        version = checkpoint.get("weights")
        if not isinstance(version, str):
            raise ValueError("В checkpoint отсутствует версия предобработки weights.")
        try:
            weights = weights_enum.IMAGENET1K_V1 if version.lower() == "none" else weights_enum[version]
        except KeyError as error:
            raise ValueError("Неизвестная версия предобработки в checkpoint.") from error
        self.preprocess = weights.transforms()
        self.model = create_model(self.name, None, len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.lock = threading.Lock()
        self.checkpoint_name = path.name
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.version = digest.hexdigest()[:12]
        self.weights = weights.name
        self.fold = checkpoint.get("fold")
        self.best_epoch = checkpoint.get("best_epoch")

    def metadata(self):
        return {
            "name": self.name,
            "device": str(self.device),
            "classes": self.classes,
            "checkpoint": self.checkpoint_name,
            "version": self.version,
            "weights": self.weights,
            "input_size": list(self.preprocess.crop_size),
            "fold": self.fold,
            "best_epoch": self.best_epoch,
        }

    def predict(self, images):
        # Один forward одновременно: запросы не раздувают память GPU.
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            probabilities = self.model(batch).float().softmax(dim=1).cpu()
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Модель вернула некорректные вероятности.")
            elapsed = round((time.perf_counter() - started) * 1000, 2)
        results = []
        for row in probabilities:
            index = int(row.argmax())
            results.append({
                "predicted_class": self.classes[index],
                "probability": float(row[index]),
                "probabilities": {name: float(value) for name, value in zip(self.classes, row)},
            })
        return results, elapsed
