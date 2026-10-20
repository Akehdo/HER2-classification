# HER2 Patch Lab

Учебная платформа для классификации готовых патчей ИГХ рака молочной железы по четырём категориям HER2: **0, 1+, 2+, 3+**. Проект включает обучение моделей PyTorch, веб-интерфейс и API предсказаний.

**Команда:** Ruslan Kadirov, Amir Alimov · **Группа:** SE-2431.

Поддерживаются **ResNet18, ResNet50 и EfficientNet-B0**. Пользователь загружает изображения, получает класс и вероятности всех классов и экспортирует результаты в CSV. При запуске выбирается один checkpoint; ансамбль фолдов не используется. Это исследовательский прототип для готовых патчей, не система клинической диагностики. Открытие WSI и автоматическая нарезка патчей не реализованы.

## Установка и запуск: выберите свою систему

Для демонстрации достаточно CPU, обученного checkpoint и папки `test_samples`. Обучать модель заново не нужно. Интернет нужен для скачивания проекта и зависимостей; сервер затем работает локально. Веса и библиотеки занимают значительное место: оставьте несколько гигабайт свободного диска.

**Выполняйте команды по очереди. Если шаг завершился ошибкой, исправьте её до следующего шага.** Не копируйте приглашения `PS C:\...>` / `$` и текст ошибок как команды. Активация `.venv` не нужна.

| Система / терминал | Инструкция |
| --- | --- |
| Windows 10/11, x64, PowerShell | Раздел Windows ниже — рекомендуемый вариант для Windows |
| Windows, Git Bash | Отдельный раздел Git Bash: пути отличаются от PowerShell |
| macOS, Apple Silicon (M1/M2/M3 и новее) | Раздел macOS; Python должен работать как `arm64` |
| macOS, Intel | Раздел Mac с Intel: текущие зависимости требуют Linux-среду или другой компьютер |
| Linux x86_64, Ubuntu/Debian/Fedora | Раздел Linux |
| Windows ARM, Linux ARM, старые ОС, Alpine Linux | Наличие подходящих сборок зависит от платформы; основной сценарий здесь не проверен. При ошибке установки используйте совместимую Linux x86_64-среду или другой поддерживаемый компьютер |

Инструкции используют **64-битный Python 3.13** для новой установки на Windows/Mac. Если подходящий Python уже установлен, переустанавливать его не нужно. На Linux можно использовать системный Python, совместимый с зависимостями (например, 3.12 или 3.13). Не устанавливайте автоматически самую новую версию Python: для неё могут ещё отсутствовать сборки библиотек. Актуальную совместимость проверяйте в [инструкции PyTorch](https://pytorch.org/get-started/locally/).

### 1. Получите проект — с Git или без него

Если Git установлен:

```text
git clone https://github.com/Akehdo/HER2-classification.git
cd HER2-classification
```

Если команда `git` не найдена, установка Git необязательна: откройте [репозиторий](https://github.com/Akehdo/HER2-classification), выберите **Code → Download ZIP**, полностью распакуйте архив и откройте терминал в распакованной папке. Она может называться `HER2-classification-master`.

**Корень проекта — папка, где одновременно лежат `app.py` и `requirements.txt`.** Родительская папка `her2` не подходит. Внутри ZIP запускать проект нельзя. Если скачивание прервалось, повторите его; размер репозитория с моделями большой.

### 2A. Windows — PowerShell

Откройте папку проекта в Проводнике, в адресной строке введите `powershell` и нажмите Enter. Проверьте файлы и Python:

```powershell
Get-Item app.py, requirements.txt
py --version
```

**Python уже есть:** если `py --version` показывает версию, используйте `py -m venv .venv` ниже. Если работает только `python --version` и он показывает настоящую версию, замените `py` на `python` в команде создания окружения.

**Python отсутствует / команды не найдены:** установите 64-битный Python 3.13 со страницы [Python для Windows](https://www.python.org/downloads/windows/). Для обычного установщика включите `Add python.exe to PATH` и установку launcher, если эти пункты предлагаются. Если установлен новый Python Install Manager, выполните `py install 3.13`. Старый launcher команду `install` не поддерживает — в таком случае нужен установщик Python. Закройте терминал, откройте заново и проверьте `py -3.13 --version`.

Если установлены несколько версий, используйте `py -3.13 -m venv .venv` вместо первой команды ниже. Список версий старого launcher: `py --list-paths`.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchvision, fastapi; print('Dependencies OK:', torch.__version__)"
.\.venv\Scripts\python.exe app.py --checkpoint "runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt" --device cpu
```

**Если `python -m venv .venv` печатает только `Python` или открывает Microsoft Store:** это может быть псевдоним Windows, а не установленный интерпретатор. Используйте рабочий `py`; проверьте `Get-Command python, py`. Если оба не работают, установите Python по инструкции выше. Не продолжайте установку пакетов, пока `.\.venv\Scripts\python.exe --version` не заработает.

В `cmd.exe` команды с `.\.venv\Scripts\python.exe` также работают; команды `Get-Item` и `Get-Command` относятся только к PowerShell.

### 2B. Windows — Git Bash

Используйте тот же установленный Windows Python. Проверьте `py --version`; если он отсутствует, выполните установку из раздела PowerShell и заново откройте Git Bash.

```bash
ls app.py requirements.txt
py -m venv .venv
./.venv/Scripts/python.exe --version
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -c "import torch, torchvision, fastapi; print('Dependencies OK:', torch.__version__)"
./.venv/Scripts/python.exe app.py --checkpoint "runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt" --device cpu
```

Здесь нужны прямые слеши `/`. PowerShell-путь `.\.venv\Scripts\python.exe` Bash прочитает неправильно. Git Bash использует Windows-окружение с `Scripts`, а WSL — Linux-окружение с `bin`: для WSL следуйте разделу Linux.

### 2C. macOS — Apple Silicon

Откройте Terminal в папке проекта. Тип процессора можно посмотреть в меню Apple → «Об этом Mac». Проверьте Python:

```bash
ls app.py requirements.txt
python3 --version
python3 -c "import platform, struct; print(platform.machine(), struct.calcsize('P') * 8)"
```

Ожидается `arm64 64`. Если на Mac с M-серией показано `x86_64`, отключите запуск Terminal через Rosetta и используйте ARM/universal2 Python. После замены Python создайте новое окружение; готовое окружение другой архитектуры не переносится.

**Python отсутствует:** установите Python 3.13 через macOS installer с [python.org](https://www.python.org/downloads/macos/), затем заново откройте Terminal. Если Homebrew уже установлен, альтернативный вариант — `brew install python@3.13`. Проверьте `python3.13 --version` и создайте окружение командой `python3.13 -m venv .venv`. При установке через Homebrew и отсутствии команды в PATH используйте `"$(brew --prefix python@3.13)/bin/python3.13" -m venv .venv`.

Если подходящий `python3` уже установлен:

```bash
python3 -m venv .venv
./.venv/bin/python --version
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python -c "import torch, torchvision, fastapi; print('Dependencies OK:', torch.__version__)"
./.venv/bin/python app.py --checkpoint "runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt" --device cpu
```

Если вы уже создали `.venv` через `python3.13`, начинайте со второй команды. CUDA на Mac не используется. Выбор Apple MPS текущим приложением не реализован, поэтому запускайте с `--device cpu`. Файлы модели `.pt` те же, что на Windows/Linux.

### 2D. Linux — Ubuntu/Debian или Fedora

В терминале проверьте:

```bash
python3 --version
ls app.py requirements.txt
```

Если Python или модуль `venv` отсутствуют, для Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

Для Fedora:

```bash
sudo dnf install python3 python3-pip
```

После установки проверьте версию. Если система предлагает слишком старый Python или версия несовместима с PyTorch, используйте поддерживаемую версию Python из репозитория вашей ОС либо современную Linux-среду. Не заменяйте системный Python вручную. Если нет прав `sudo`, попросите администратора установить Python/venv или используйте компьютер, где они уже доступны.

Создайте окружение и запустите приложение:

```bash
python3 -m venv .venv
./.venv/bin/python --version
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python -c "import torch, torchvision, fastapi; print('Dependencies OK:', torch.__version__)"
./.venv/bin/python app.py --checkpoint "runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt" --device cpu
```

Не используйте `sudo pip install`. Зависимости устанавливаются в `.venv`, а не в системный Python. Если выбрали отдельную версию, например `python3.13`, используйте её в команде создания окружения.

### 2E. Mac с Intel

Нативная установка текущего проекта через `requirements.txt` не поддерживается готовыми официальными сборками: проект требует `torch>=2.6`, а [PyTorch завершил поддержку macOS x86_64 на версии 2.2.x](https://pytorch.org/blog/pytorch2-2/). Простая замена команды `python` или запуск на CPU это не исправят. Снижать требование до старого torch без адаптации и проверки проекта не следует.

Варианты запуска:

1. Использовать Windows/Linux x86_64 или Mac с Apple Silicon по инструкциям выше.
2. На Intel Mac создать виртуальную машину Ubuntu x86_64 в совместимой программе виртуализации, скопировать/скачать проект **внутри Linux**, установить Python по разделу Linux и открыть браузер внутри этой виртуальной машины. Старое `.venv` с macOS туда не копировать. Вариант требует ресурсов для гостевой ОС и отдельно здесь не проверялся.

### 3. Убедитесь, что запуск успешен

После старта оставьте терминал открытым и перейдите в браузере на **http://127.0.0.1:8000**. Адрес открывается на том компьютере, где запущен сервер (для VM — внутри VM).

- В терминале должен появиться адрес сервера без завершающей ошибки.
- По адресу **http://127.0.0.1:8000/health** поле `status` должно быть `ready`.
- Загрузите 1–2 изображения из `test_samples`, получите предсказания и проверьте экспорт CSV.

Открытие страницы само по себе не доказывает загрузку модели. При `not_ready` проверьте checkpoint и ошибку в терминале. Завершение сервера: **Ctrl+C**. Для повторного запуска установка зависимостей не нужна — откройте папку проекта и повторите только команду `app.py` для своей ОС.

Все приведённые команды используют CPU. На Windows/Linux можно подобрать сборку без CUDA через [официальный установщик PyTorch](https://pytorch.org/get-started/locally/) (OS → Pip → CPU), выполняя установку через Python из `.venv`, затем повторить установку `requirements.txt` и `pip check`.

### Выбор модели

| Модель | Пример checkpoint |
| --- | --- |
| ResNet18 | `runs/resnet18_IMAGENET1K_V1_seed32/fold_1.pt` |
| ResNet50 | `runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt` |
| EfficientNet-B0 | `runs/efficientnet_b0_IMAGENET1K_V1_seed32/fold_1.pt` |

Для смены модели перезапустите сервер с другим `--checkpoint`. Архитектура и порядок классов считываются из файла. Нужен checkpoint формата `main.py`, включающий `model`, `classes`, `weights` и `model_state_dict`; обычных ImageNet-весов недостаточно.

- `--device cpu` — процессор.
- `--device auto` — CUDA при наличии, иначе CPU.
- `--device cuda` — NVIDIA GPU; нужен совместимый PyTorch с CUDA.
- `--port 8001` — другой порт, если 8000 занят.

Также поддерживаются переменные `HER2_CHECKPOINT` и `HER2_DEVICE`. Если путь не задан ни аргументом, ни переменной, приложение ищет `models/her2.pt`. Обучение при запуске сервера не выполняется.

## Проверка платформы на примерах

В `test_samples` находятся **120 изображений: по 30 каждого класса**. Для демонстрации полный датасет не требуется.

| Папка | Исходная метка | Количество |
| --- | --- | ---: |
| `test_samples/class_0` | 0 | 30 |
| `test_samples/class_1+` | 1+ | 30 |
| `test_samples/class_2+` | 2+ | 30 |
| `test_samples/class_3+` | 3+ | 30 |

1. Запустите сервер с checkpoint и откройте главную страницу.
2. Убедитесь, что модель готова. По адресу http://127.0.0.1:8000/health поле `status` должно быть `ready`.
3. Загрузите несколько изображений из одной или разных папок `test_samples`.
4. Запустите обработку. Для каждого файла должны появиться предсказанный класс и вероятности четырёх классов.
5. Сравните предсказания с метками папок. Ошибка классификации сама по себе не означает поломку приложения.
6. Экспортируйте результаты в CSV и проверьте соответствие файлов и предсказаний.
7. Для сравнения моделей перезапустите сервер с другим checkpoint и загрузите те же изображения.

**Ограничения загрузки:** до 16 файлов за запрос, до 10 МБ на файл и до 32 МБ на весь запрос. Форматы: PNG, JPEG, WebP, одностраничный TIFF; до 16 миллионов пикселей на изображение. Поэтому 30 примеров класса нужно отправлять несколькими партиями.

Названия папок помогают проверить результат вручную; правильная метка из имени папки модели не передаётся. Вероятность softmax не гарантирует правильность ответа. Этот набор служит для демонстрации и не заменяет оценку качества на независимом полном тестовом наборе.

## Автоматические тесты

Установите зависимости для разработки, включающие основные зависимости. **Windows PowerShell/cmd:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Только проверки веб-приложения:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_web.py" -v
```

**macOS/Linux:**

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python -m unittest discover -s tests -p "test_web.py" -v
```

**Windows Git Bash:**

```bash
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe -m unittest discover -s tests -p "test_web.py" -v
```

Последняя команда в каждом блоке повторяет только веб-тесты: после успешного полного запуска повторять её необязательно.

Отдельный запущенный сервер не требуется. Успешный запуск завершается `OK`; ошибки показываются как `FAIL` или `ERROR`.

| Файл | Что проверяется |
| --- | --- |
| `tests/test_web.py` | Загрузка архитектур, preprocessing и инференс, API, метаданные, лимиты, некорректные изображения и недоступный checkpoint |
| `tests/test_models.py` | Создание моделей, классификатор, градиенты и обучение на синтетических данных |
| `tests/test_weights.py` | Варианты pretrained-весов и их загрузка |
| `tests/test_metrics.py` | Метрики, граничные случаи и экспорт результатов |
| `tests/test_early_stopping.py` | Ранняя остановка и сохранение результатов лучшей эпохи |

Веб-тесты создают временные checkpoint со случайными весами. Они проверяют код, не качество обученных моделей, и не требуют готовых checkpoint из `runs`.

## Архитектура

Поток обработки: **браузер → FastAPI → декодирование изображения → preprocessing torchvision → модель PyTorch → softmax → JSON → результаты в браузере**.

| Компонент | Назначение |
| --- | --- |
| `static/index.html`, `static/styles.css`, `static/app.js` | Интерфейс на HTML/CSS/JavaScript: загрузка, предпросмотр, результаты и CSV. Сборка Node.js не требуется |
| `app.py` | FastAPI/Uvicorn, раздача интерфейса, загрузка модели при старте, API и ограничения запросов |
| `inference.py` | Проверка изображений, загрузка checkpoint, CPU/CUDA, preprocessing и предсказания |
| `model_registry.py` | Общий реестр архитектур для обучения и инференса |
| `main.py` | Обучение, кросс-валидация, ранняя остановка, checkpoint и метрики |
| `tests/` | Автоматические проверки |
| `test_samples/` | Изображения для ручной демонстрации |
| `runs/` | Веса и отчёты запусков обучения |
| `Patch-based-dataset/` | Полный локальный датасет, исключённый из Git |

Модель загружается один раз при старте. Изображения приводятся к RGB и обрабатываются преобразованиями torchvision для версии весов из checkpoint. Инференс выполняется в режиме оценки без вычисления градиентов.

Базы данных нет. Сервер не сохраняет загруженные изображения на диск. Результаты находятся в браузере и могут быть экспортированы пользователем в CSV; обновление страницы сбрасывает состояние интерфейса.

## HTTP API

Интерактивная документация: http://127.0.0.1:8000/docs. Для загрузки Swagger может потребоваться интернет.

| Метод и адрес | Назначение |
| --- | --- |
| `GET /` | Веб-интерфейс |
| `GET /health` | Готовность, метаданные модели и ограничения. Проверяйте поле `status`: HTTP 200 сам по себе не означает готовность |
| `POST /predict` | Изображения в `multipart/form-data`, поле `files`; ответ содержит модель, время обработки и результаты по файлам |

Пример PowerShell при работающем сервере:

```powershell
$sample = Get-ChildItem -LiteralPath ".\test_samples\class_0" -Filter *.png | Select-Object -First 1
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "files=@$($sample.FullName)"
```

Основные ошибки: `503` — модель недоступна; `413` — превышен размер; `415` — неверный тип запроса; `422` — пустая загрузка или недопустимое изображение. Превышение числа файлов может возвращать `400`. Некорректный файл отклоняет весь запрос. `inference_ms` не включает передачу файлов по сети.

## Обучение и оценка качества

Обучение запускается отдельно. Нужен полный обучающий набор HER2-IHC-40x с подпапками классов:

```text
Patch-based-dataset/
  train_data_patch/
    class_0/
    class_1+/
    class_2+/
    class_3+/
```

Пример нового эксперимента на CPU (эти параметры не являются утверждением о настройках всех существующих запусков):

```powershell
.\.venv\Scripts\python.exe main.py --data "Patch-based-dataset\train_data_patch" --model resnet18 --weights IMAGENET1K_V1 --models-dir "models" --seed 32 --folds 5 --epochs 50 --patience 5 --batch-size 16 --device cpu --output "runs\new_experiment"
```

Команда выше написана для Windows PowerShell. На macOS/Linux замените исполняемый файл на `./.venv/bin/python` и обратные слеши в путях на `/`; в Git Bash используйте `./.venv/Scripts/python.exe` и `/`.

На CPU обучение может занимать продолжительное время. Для GPU используйте `--device cuda` при наличии совместимого PyTorch. Pretrained-веса должны быть доступны локально или для загрузки. Все параметры: `main.py --help`.

Результаты включают `fold_*.pt`, `history.csv`, `metrics.csv`, отчёты по классам, матрицы ошибок и `oof_predictions.csv`. Для новых экспериментов задавайте отдельный `--output`, чтобы не перезаписывать запуск с теми же моделью, весами и seed. `test_samples` не следует использовать как обучающий набор.

Основная метрика исследовательского сравнения — **macro-F1**, одинаково учитывающая каждый класс. Дополнительно следует анализировать метрики отдельных классов, матрицу ошибок и сравнивать с majority-class baseline — предсказанием наиболее частого класса обучающей выборки.

**Ограничение реализации:** `main.py` использует `StratifiedKFold` на уровне патчей, а не группировку по WSI. Патчи одного стекла могут оказаться в разных частях выборки. Для независимого WSI-level split нужны идентификаторы исходных стёкол и групповое разделение; текущие результаты нельзя представлять как такую оценку.

Validation loss используется для выбора лучшей эпохи, поэтому OOF-результаты на этих же фолдах не заменяют финальный независимый тест. Сравнение архитектур требует одинаковых разбиений и сопоставимых бюджетов обучения. Подбор параметров по тесту и выбор только удачных запусков искажают оценку. Один датасет не доказывает переносимость на другие лаборатории.

## Если запуск не удался

| Симптом | Решение |
| --- | --- |
| `python` выводит только `Python` или открывает Store | Windows: проверьте `py --version`, создайте окружение через `py`. Если нет рабочего Python, установите его по разделу Windows |
| `py` / `python3` не найден | Установите Python для своей ОС, закройте и откройте терминал; проверьте версию до создания `.venv` |
| `No suitable Python runtime found` | Запрошенная версия не установлена. Установите её либо выберите уже установленную совместимую версию |
| `.venv/.../python` не найден | Создание окружения не завершилось или открыта другая папка. Проверьте корень проекта, повторите создание и проверку версии |
| `..venvScriptspython.exe: command not found` | Вы в Git Bash. Используйте `./.venv/Scripts/python.exe`, а не путь PowerShell |
| `requirements.txt` / `app.py` не найден | Перейдите в распакованную папку, где действительно находятся эти файлы, не в родительскую папку |
| `ensurepip is not available` / нет `venv` | Ubuntu/Debian: установите `python3-venv`; для отдельно установленного Python нужен соответствующий пакет venv этой версии |
| `externally-managed-environment` | Вы устанавливаете пакеты в системный Python. Создайте `.venv` и используйте его Python; не применяйте `--break-system-packages` |
| PowerShell запрещает выполнение `Activate.ps1` | Активация не требуется. Используйте прямой путь `.\.venv\Scripts\python.exe`; менять ExecutionPolicy не нужно |
| `No matching distribution found` для torch/torchvision | Проверьте версию Python, разрядность и архитектуру. Используйте совместимый 64-битный Python; для Intel Mac см. отдельный раздел. Если версия Python слишком новая, создайте окружение на 3.13. Проверьте также доступ к индексу пакетов |
| `DLL load failed` / `WinError 126` | Windows: проверьте 64-битный Python и согласованность torch/torchvision. Если ошибка указывает на отсутствующий runtime, установите [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) и повторите импорт |
| `operator torchvision::nms does not exist` | Возможна несовместимая пара torch/torchvision. В новом окружении установите их совместно по официальной инструкции PyTorch, затем зависимости проекта |
| `ModuleNotFoundError` | Установка не закончилась либо сервер запускается другим Python. Используйте один и тот же Python из `.venv` для pip и app.py |
| SSL / certificate verify failed | Проверьте дату компьютера и сертификаты сети. На macOS с Python с python.org запустите `Install Certificates.command` из папки установленного Python, если он присутствует. Не отключайте проверку TLS |
| Timeout / connection failed при установке | Проверьте интернет, прокси и доступ к PyPI/серверам PyTorch; повторите неудавшуюся команду после исправления сети |
| `Permission denied` | Распакуйте проект в доступную для записи пользовательскую папку. Не устанавливайте пакеты через sudo |
| Не хватает места | Освободите место для проекта, checkpoint, окружения и загрузок pip; затем повторите установку |
| Сервер работает, но `/health` показывает `not_ready` | Проверьте существование `.pt`, полный размер файла и текст ошибки загрузки. Интерфейс может открываться даже без модели |
| `CUDA is not available` | Для демонстрации используйте `--device cpu`. На Mac CUDA не используется |
| `Address already in use` / WinError 10048 | Добавьте `--port 8001` и откройте http://127.0.0.1:8001 |
| Браузер пишет «не удаётся подключиться» | Убедитесь, что сервер не завершился, порт совпадает и браузер открыт на том же компьютере. Используйте `http`, не `https` |
| Обработка медленная / процесс закрывается из-за памяти | Отправляйте по одному изображению, закройте тяжёлые программы; попробуйте checkpoint ResNet18. CPU работает медленнее GPU |
| Изображение отклонено | Проверьте формат, размер, разрешение, одностраничность TIFF и лимиты загрузки |

### Проверка checkpoint

Файл весов должен быть настоящим checkpoint, а не незавершённой загрузкой или текстовым Git LFS pointer. При ошибке формата повторно получите исходный файл. Не переименовывайте произвольные веса в `fold_1.pt`.

PowerShell:

```powershell
Get-Item "runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt" | Select-Object FullName, Length
```

macOS/Linux/Git Bash:

```bash
ls -lh "runs/resnet50_IMAGENET1K_V1_seed32/fold_1.pt"
```

Если файл отсутствует, проверьте папку `runs` и укажите фактический путь через `--checkpoint`. Получите недостающие веса у авторов; загрузка случайных pretrained ImageNet-весов не заменяет HER2 checkpoint.

### Как пересоздать нерабочее окружение

После смены ОС, архитектуры или версии Python старое окружение повторно использовать нельзя. Остановите сервер. Если окружение было активировано, выполните `deactivate` или откройте новый терминал. Сохраните старую папку под свободным именем, например `.venv-backup`:

PowerShell:

```powershell
Rename-Item -LiteralPath .venv -NewName .venv-backup
```

macOS/Linux/Git Bash:

```bash
mv .venv .venv-backup
```

Если такое имя уже занято, выберите другое. Затем повторите создание `.venv` выбранным Python и установку зависимостей по инструкции своей ОС. Не переносите `.venv` между компьютерами.

### Что отправить автору, если ошибка осталась

Укажите ОС, тип процессора, версию Python и полный текст ошибки, начиная с введённой команды. Скриншота последней строки часто недостаточно. Добавьте вывод диагностики (она не требует установленного torch):

PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import sys, platform, struct; print(sys.executable); print(sys.version); print(platform.platform()); print(platform.machine(), struct.calcsize('P') * 8)"
.\.venv\Scripts\python.exe -m pip check
```

macOS/Linux:

```bash
./.venv/bin/python -c "import sys, platform, struct; print(sys.executable); print(sys.version); print(platform.platform()); print(platform.machine(), struct.calcsize('P') * 8)"
./.venv/bin/python -m pip check
```

Для Git Bash замените путь на `./.venv/Scripts/python.exe`. Если окружение ещё не создалось, выполните диагностику через рабочий `py` / `python3` вместо пути к `.venv` и сообщите ошибку создания окружения.

Инструкции сверены с кодом приложения и официальной документацией. Полный запуск на каждой ОС и конфигурации оборудования не проверялся; зависимости в `requirements.txt` задают диапазоны, а не фиксированное окружение. Наличие инструкции не гарантирует поддержку любой старой ОС или процессора.
