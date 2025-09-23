# Установка

> :material-rocket-launch-outline: Установите TensorAeroSpace за 10 секунд и начните эксперименты.

!!! note
    Поддерживаемые версии Python: 3.9 — 3.11. Поддержка Python 3.12 запланирована.

| :material-python: Python | Статус |
|-------------------------:|:------:|
| 3.9                      | ✅ |
| 3.10                     | ✅ |
| 3.11                     | ✅ |
| 3.12                     | ⏳ |

## Быстрая установка (PyPI)

=== "pip"

    ```bash
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

=== "poetry"

    ```bash
    poetry add tensoraerospace
    ```

=== "conda"

    ```bash
    conda create -n tas python=3.10 -y
    conda activate tas
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

## Проверка установки

Быстрая проверка версии и минимального примера:

```bash
python -c "import tensoraerospace as tas; print(tas.__version__)"
```

```python
import gymnasium as gym
from tensoraerospace.envs import LinearLongitudinalF16

env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
state, _ = env.reset()
for _ in range(256):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Платформы и окружение

<div class="grid cards" markdown>

-   :material-linux: **Linux**

    Рекомендуемая платформа для обучения. Достаточно `pip` и совместимых колёс.

-   :material-microsoft-windows: **Windows**

    Работает нативно; для продвинутых сценариев рекомендуем WSL2.

-   :material-apple: **macOS (Intel/M‑series)**

    На Apple Silicon используйте совместимые версии Python (3.9–3.11) и `tensorflow-macos`.

</div>

## Установка из исходников (Dev)

=== "Poetry"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace
    poetry install
    ```

=== "pip + venv"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    pip install -e .
    ```

!!! tip
    Для изоляции зависимостей всегда используйте виртуальное окружение (`venv`/`conda`/`poetry`). Это предотвратит конфликты с глобальными пакетами.

## Подсказки по CPU/GPU

- В проекте используются и TensorFlow, и PyTorch. Если вам нужна поддержка GPU, устанавливайте фреймворки с учётом вашей платформы:
  - PyTorch с CUDA: следуйте официальной инструкции по установке версии, совместимой с вашей CUDA.
  - macOS (Apple Silicon): пакет `tensorflow-macos` устанавливается автоматически (см. зависимости проекта), убедитесь, что используете совместимую версию Python (3.9–3.11).
- Если не требуется GPU‑ускорение, стандартные колёса PyPI обычно достаточно.

## Запуск через Docker

!!! info
    :material-docker: Docker — рекомендуемый способ для унифицированного окружения на Linux/Windows/macOS.

Сборка образа:

```bash
docker build -t tensoraerospace .
```

Запуск контейнера с пробросом примеров и порта Jupyter (если потребуется):

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)/example:/app/example" \
  --name tas tensoraerospace
```

!!! tip
    Пробрасывайте нужные директории через `-v <host>:<container>`, чтобы сохранять результаты вне контейнера.

## Частые проблемы и решения

???+ question "Не удаётся разрешить зависимости"
    Чаще всего помогает изоляция окружения и обновление инструментов сборки.

    === "venv"

        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
        pip install -U pip setuptools wheel
        pip install tensoraerospace
        ```

    === "conda"

        ```bash
        conda create -n tas python=3.10 -y
        conda activate tas
        pip install -U pip setuptools wheel
        pip install tensoraerospace
        ```

??? question "Конфликт версий TensorFlow/PyTorch"
    Установите версии под вашу платформу (CUDA/CPU/macOS), затем повторите установку `tensoraerospace`.

??? question "macOS (M‑серия)"
    Используйте Python 3.9–3.11. При необходимости переустановите `tensorflow-macos` совместимой версии.

??? question "Проблемы с правами/ сетью"
    Пробуйте в чистом виртуальном окружении или внутри Docker. Для корпоративных прокси настройте `PIP_*` переменные окружения.

## Следующие шаги

[:material-play-circle-outline: К примерам](../example/env/examples.md){ .md-button .md-button--primary }
[:material-airplane-takeoff: Модели](../model/f16.md){ .md-button }
[:material-robot-outline: Алгоритмы](../agent/sac.md){ .md-button }
[:material-book-open-variant: Учебные уроки](../lesson/0intro.md){ .md-button }
