"""Локальный веб-интерфейс и HTTP API для классификации готовых патчей."""

import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from inference import InvalidImage, Predictor, decode_image


ROOT = Path(__file__).resolve().parent
MAX_FILES = 16
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
logger = logging.getLogger("her2")


class RequestLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared_size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Некорректный Content-Length."}, 400)(scope, receive, send)
        if declared_size > MAX_REQUEST_BYTES:
            return await JSONResponse({"detail": "Общий размер загрузки превышает 32 МБ."}, 413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Общий размер загрузки превышает 32 МБ.")
            return message

        async def secure_send(message):
            if message["type"] == "http.response.start":
                policy = (b"default-src 'self'; img-src 'self' blob: data:; "
                          b"style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
                          b"object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
                if scope["path"] == "/docs":
                    # Встроенный Swagger UI FastAPI использует официальный CDN и inline bootstrap.
                    policy = (b"default-src 'self'; img-src 'self' data: https://fastapi.tiangolo.com; "
                              b"style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                              b"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                              b"connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
                message.setdefault("headers", []).extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"same-origin"),
                    (b"cache-control", b"no-store"),
                    (b"content-security-policy", policy),
                ])
            await send(message)

        await self.app(scope, limited_receive, secure_send)


def create_app(checkpoint=None, device=None):
    selected = checkpoint or os.getenv("HER2_CHECKPOINT")
    if not selected and (ROOT / "models" / "her2.pt").is_file():
        selected = ROOT / "models" / "her2.pt"
    selected_device = device or os.getenv("HER2_DEVICE", "auto")

    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = None
        application.state.model_error = None
        if selected:
            path = Path(selected)
            if not path.is_absolute():
                path = ROOT / path
            try:
                application.state.predictor = await run_in_threadpool(Predictor, path, selected_device)
            except Exception:
                logger.exception("Не удалось загрузить checkpoint")
                application.state.model_error = (
                    "Не удалось загрузить checkpoint. Проверьте путь, формат весов и устройство; "
                    "подробности есть в терминале сервера.")
        yield
        application.state.predictor = None

    application = FastAPI(
        title="HER2 Patch Lab", version="1.0.0", lifespan=lifespan,
        description="Классификация готовых патчей с помощью обученных моделей PyTorch.",
        docs_url="/docs", redoc_url=None, swagger_ui_parameters={"syntaxHighlight": False},
    )
    application.add_middleware(RequestLimitMiddleware)
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "static" / "index.html")

    @application.get("/health")
    async def health(request: Request):
        predictor = request.app.state.predictor
        return {
            "status": "ready" if predictor else "not_ready",
            "model": predictor.metadata() if predictor else None,
            "message": request.app.state.model_error or (
                None if predictor else "Подключите обученный checkpoint и перезапустите сервер."),
            "limits": {"files": MAX_FILES, "file_mb": 10, "request_mb": 32},
        }

    @application.post("/predict", openapi_extra={"requestBody": {
        "required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["files"], "properties": {
                "files": {"type": "array", "minItems": 1, "maxItems": MAX_FILES,
                          "items": {"type": "string", "format": "binary"}},
            },
        }}},
    }})
    async def predict(request: Request):
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, request.app.state.model_error or
                                "Модель не подключена. Укажите --checkpoint и перезапустите сервер.")
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            raise HTTPException(415, "Отправьте изображения как multipart/form-data в поле files.")
        async with request.form(max_files=MAX_FILES, max_fields=0) as form:
            files = form.getlist("files")
            if not files or any(not isinstance(file, UploadFile) for file in files):
                raise HTTPException(422, "Добавьте от 1 до 16 изображений в поле files.")
            if len(files) != len(form.multi_items()):
                raise HTTPException(422, "Все изображения должны находиться в поле files.")
            images, filenames = [], []
            for file in files:
                # Читаем не больше лимита даже при отсутствующем/ложном Content-Length.
                data = await file.read(MAX_FILE_BYTES + 1)
                filename = (file.filename or "patch").replace("\\", "/").split("/")[-1][:200]
                if len(data) > MAX_FILE_BYTES:
                    raise HTTPException(413, f"Файл {filename}: максимум 10 МБ.")
                try:
                    image = await run_in_threadpool(decode_image, data)
                except InvalidImage as error:
                    raise HTTPException(422, f"Файл {filename}: {error}") from error
                images.append(image)
                filenames.append(filename)
            try:
                results, elapsed = await run_in_threadpool(predictor.predict, images)
            except Exception:
                logger.exception("Ошибка инференса")
                raise HTTPException(500, "Не удалось выполнить инференс. Подробности в терминале сервера.")
            return {
                "model": predictor.metadata(),
                "inference_ms": elapsed,
                "results": [{"filename": name, "width": image.width, "height": image.height, **result}
                            for name, image, result in zip(filenames, images, results)],
            }

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Обученный fold_*.pt из main.py")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint, args.device), host=args.host, port=args.port, workers=1)
