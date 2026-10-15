# HER2: ResNet18 и ResNet50

Простой baseline на PyTorch: ImageNet-веса → замена последнего слоя → дообучение всей сети. Весь код обучения находится в `main.py`.

## Установка и данные

Команды для PowerShell из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Expand-Archive -LiteralPath .\Patch-based-dataset\train_data_patch.zip -DestinationPath .\Patch-based-dataset\train_data_patch
```

Распаковать архив нужно один раз. Ожидаемая структура:

```text
Patch-based-dataset/
  train_data_patch/
    class_0/
    class_1+/
    class_2+/
    class_3+/
  test_data_patch.zip
```

Если изображения лежат в другом месте, укажите `--data "D:\data\train"`. Внутри должны находиться папки классов с изображениями.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py --model resnet18 --weights IMAGENET1K_V1 --seed 42 --folds 5 --epochs 10
.\.venv\Scripts\python.exe main.py --model resnet50 --weights IMAGENET1K_V1 --seed 42 --folds 5 --epochs 10
```

Дополнительно можно менять `--batch-size`, `--workers`, `--lr 0.0001` и `--output runs`. Без явных настроек на CUDA выбираются batch size 256 и до 8 процессов загрузки, на CPU — batch size 16 и 0 процессов. Для небольшой GPU ноутбука задайте `--batch-size 8` или `16`.

## Сервер с одной H200

На сервере нужен PyTorch с поддержкой CUDA. Если он ещё не установлен, выберите CUDA-сборку под драйвер сервера на [сайте PyTorch](https://pytorch.org/get-started/locally/), затем установите остальные зависимости из `requirements.txt`. Копировать Windows-папку `.venv` на Linux не нужно.

Проверка окружения:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
```

Команды из корня проекта на сервере:

```bash
python main.py --model resnet18 --weights IMAGENET1K_V1 --device cuda --precision bf16 --batch-size 256 --workers 8 --seed 42 --folds 5 --epochs 10
python main.py --model resnet50 --weights IMAGENET1K_V1 --device cuda --precision bf16 --batch-size 256 --workers 8 --seed 42 --folds 5 --epochs 10
```

`--device cuda` выдаёт понятную ошибку, если CUDA недоступна. В начале запуска печатаются название GPU и фактические настройки; после каждой эпохи — её время, включая validation.

Для CUDA включены BF16 mixed precision (при `--precision bf16`), `channels_last`, fused Adam, подбор алгоритмов cuDNN и TF32 для оставшихся FP32-матричных операций. Загрузка данных использует отдельные процессы, pinned memory, предварительную загрузку и persistent workers. Метрики переносятся на CPU один раз за эпоху. Это стандартные приёмы из [руководства PyTorch по ускорению](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

256 — начальное значение batch size для H200, а не результат замера на вашем сервере. Если сервер выделяет меньше 8 CPU-ядер, уменьшите `--workers`. Храните распакованные изображения на локальном SSD сервера: скорость декодирования PNG и диска тоже влияет на время обучения. Для сравнения моделей используйте одинаковые batch size, seed и число эпох; learning rate автоматически не меняется. При изменении batch size качество стоит сравнивать по CV.

Для строгой воспроизводимости добавьте `--deterministic`: это отключает cuDNN benchmark и включает детерминированные алгоритмы. Быстрый режим по умолчанию сохраняет seed для разбиения, инициализации, перемешивания и аугментаций, но допускает небольшие различия CUDA-результатов между повторными запусками ([документация PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html)). Для FP32 задайте `--precision fp32`.

Веса:

| Модель | `--weights` |
| --- | --- |
| resnet18 | IMAGENET1K_V1, DEFAULT |
| resnet50 | IMAGENET1K_V1, IMAGENET1K_V2, DEFAULT |

`DEFAULT` означает V1 для ResNet18 и V2 для ResNet50 ([документация torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)). Веса загружаются при первом запуске, затем берутся из кеша `<output>/_cache/`. `--weights none` создаёт модель со случайными весами для проверки кода.

Без `--device` CUDA используется автоматически, если доступна; иначе — CPU с FP32. Для явного выбора есть `--device cuda` и `--device cpu`. Если CUDA GPU не поддерживает BF16, используйте `--precision fp32`.

## Как устроена кросс-валидация

- `StratifiedKFold` делит **только train** на K фолдов с сохранением пропорций классов. Готовый test не участвует в обучении и выборе модели.
- Каждый фолд начинает обучение заново с выбранных ImageNet-весов. Все слои обучаются.
- В train используются случайные отражения после уменьшения изображения; в validation — только предобработка выбранных весов, включая ImageNet-нормализацию и crop 224 × 224.
- Одинаковые `--seed`, данные и число фолдов дают одинаковые разбиения для обеих моделей. Seed также задаёт инициализацию, перемешивание и аугментации, включая процессы загрузки. Для строгой воспроизводимости используйте `--deterministic`; держите одинаковыми также workers, batch size, устройство и версии библиотек.
- После каждой эпохи на validation считаются loss, accuracy, macro F1, balanced accuracy, quadratic weighted kappa и macro ROC-AUC one-vs-rest. Итог CV — среднее и стандартное отклонение метрик **последней эпохи** по фолдам.
- Balanced accuracy — среднее recall по классам. Quadratic weighted kappa учитывает расстояние между классами в порядке `ImageFolder`: для нашего датасета `class_0`, `class_1+`, `class_2+`, `class_3+`. Если использовать другие классы, этот порядок должен соответствовать их порядковой шкале.
- Дополнительно собираются OOF (out-of-fold) предсказания: каждый патч оценивается один раз моделью своего validation-фолда, которая не обучалась на этом патче. По ним строятся общий отчёт и матрицы ошибок. OOF-метрики вычисляются по всем предсказаниям вместе и могут отличаться от среднего по фолдам; OOF loss — среднее fold loss с весами по размерам фолдов.

В этом архиве имена вида `her2-0-score_test_1168.png` не содержат явного ID пациента/WSI. Поэтому здесь CV **по патчам**: код не гарантирует разделение по пациентам или стёклам. Для такой оценки потребуется таблица соответствия патчей пациентам/WSI. Разновидности исходного датасета описаны в [репозитории авторов](https://github.com/seraju77/HER2-IHC-40x-data-preprocessing).

## Результаты

Например, `runs/resnet18_IMAGENET1K_V1_seed42/` содержит:

- `metrics.csv` — `loss`, `accuracy`, `macro_f1`, `balanced_accuracy`, `quadratic_kappa`, `roc_auc_ovr_macro` последней эпохи каждого фолда, затем строки `mean`, `std` (по фолдам, `ddof=0`) и `oof`;
- `history.csv` — train loss, все validation-метрики и время каждой эпохи с номерами фолда и эпохи;
- `fold_1_classification_report.csv` … `fold_5_classification_report.csv` — precision, recall, F1, число примеров (`support`) и ROC-AUC one-vs-rest для каждого класса;
- `fold_1_confusion_matrix.csv` / `.png` и аналогичные файлы остальных фолдов — матрицы ошибок: строки — истинные классы, столбцы — предсказанные;
- `fold_1_confusion_matrix_normalized.csv` / `.png` и аналогичные файлы остальных фолдов — матрицы с нормализацией по строкам: CSV содержит доли 0–1, PNG — проценты;
- `oof_classification_report.csv`, `oof_confusion_matrix.csv` / `.png`, `oof_confusion_matrix_normalized.csv` / `.png` — общие отчёт и матрицы по OOF-предсказаниям;
- `oof_predictions.csv` — индекс и относительный путь патча, номер validation-фолда, истинный и предсказанный классы, вероятности всех классов;
- `fold_1.pt` … `fold_5.pt` — веса последней эпохи, названия классов и параметры обучения.

ROC-AUC считается по вероятностям softmax в FP32. Если для класса нет положительных или отрицательных примеров, его ROC-AUC и macro ROC-AUC записываются как `nan`; разбиение с проверкой числа примеров в текущем скрипте обеспечивает присутствие всех классов в каждом validation-фолде. Kappa также записывается как `nan`, если истинные и предсказанные метки все совпадают с одним классом. PNG строятся без графического интерфейса и подходят для запуска на сервере.

Повторный запуск с теми же model, weights и seed перезаписывает одноимённые файлы; для отдельного эксперимента задайте другой `--output`. Полное обучение запускается командами выше; отдельной оценки готового test в этом минимальном скрипте нет.
