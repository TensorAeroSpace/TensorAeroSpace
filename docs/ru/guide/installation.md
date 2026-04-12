# Установка

> :material-rocket-launch-outline: Установите TensorAeroSpace за 10 секунд и начинайте экспериментировать.

!!! note
    Поддерживаемые версии Python: 3.10 — 3.12.

| :material-python: Python | Статус |
|-------------------------:|:------:|

| 3.10                     | ✅ |
| 3.11                     | ✅ |
| 3.12                     | ✅ |

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

Быстрая проверка версии и минимальный пример:

```bash
python -c "import tensoraerospace as tas; print(tas.__version__)"
```

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.pid import PID
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

# Параметры симуляции
dt = 0.01
tp = generate_time_period(tn=10, dt=dt)  # 10 секунд
N = len(tp)

# Опорный сигнал (ступенька 5° в радианах)
reference = unit_step(degree=5, tp=tp, time_step=100, output_rad=True).reshape(1, -1)

# Создание среды F-16 (порядок состояний: [alpha, q])
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=N,
    initial_state=[[0], [0]],
    reference_signal=reference,
    use_reward=False,
)

# ПИД-контроллер (коэффициенты из примера PID)
pid = PID(env, kp=-14.290139135229715, ki=-8.240470780203491, kd=-1.2991634935096958, dt=dt)

obs, info = env.reset()
for t in range(N - 1):
    setpoint = reference[0, t]
    alpha = float(obs[0])  # env возвращает [alpha, q]
    u = pid.select_action(setpoint, alpha)
    action = np.array([[float(u)]], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Платформы и окружения

<div class="grid cards" markdown>

-   :material-linux: **Linux**

    Рекомендуемая платформа для обучения. Достаточно `pip` и совместимых колёс (wheels).

-   :material-microsoft-windows: **Windows**

    Работает нативно; для продвинутых сценариев можно использовать WSL2.

-   :material-apple: **macOS (Intel/M‑series)**

    На Apple Silicon используйте совместимые версии Python (3.10–3.12) и пакет `tensorflow-macos`.

</div>

## Установка из исходников (для разработки)

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
    Всегда изолируйте зависимости виртуальной средой (`venv`/`conda`/`poetry`), чтобы избежать конфликтов с глобальными пакетами.

## Советы по CPU/GPU

- В проекте используются TensorFlow и PyTorch. Если нужна поддержка GPU, установите фреймворки под вашу платформу:
  - PyTorch с CUDA: следуйте официальной инструкции установки, совместимой с вашей версией CUDA.
  - macOS (Apple Silicon): пакет `tensorflow-macos` устанавливается автоматически (см. зависимости проекта). Убедитесь, что версия Python совместима (3.10–3.12).
- Если ускорение на GPU не требуется, обычно достаточно стандартных колёс PyPI.

## Запуск в Docker

!!! info
    :material-docker: Docker — рекомендуемый способ получить единое окружение на Linux/Windows/macOS.

Сборка образа:

=== "Ubuntu / Linux (bash)"

    ```bash
    docker build -t tensoraerospace . --platform=linux/amd64
    ```

=== "Windows (PowerShell)"

    ```powershell
    docker build -t tensoraerospace . --platform=linux/amd64
    ```

Запустите контейнер (образ **по умолчанию поднимает JupyterLab**) и примонтируйте директорию с примерами:

=== "Ubuntu / Linux (bash)"

    ```bash
    docker run --rm -it -p 8888:8888 \
      -v "$(pwd)/example:/app/example" \
      --name tas tensoraerospace
    ```

=== "Windows (PowerShell)"

    ```powershell
    docker run --rm -it -p 8888:8888 `
      -v "${PWD}\example:/app/example" `
      --name tas tensoraerospace
    ```

!!! tip
    Чтобы включить NVIDIA GPU внутри контейнера, добавьте `--gpus all` (на Ubuntu/Linux требуется NVIDIA Container Toolkit; на Windows — Docker Desktop + поддержка GPU в WSL2).

!!! tip
    Примонтируйте нужные директории флагом `-v <host>:<container>`, чтобы сохранять результаты вне контейнера.

## Типовые проблемы и решения

???+ question "Не удаётся разрешить зависимости"
    Как правило, помогает изоляция окружения и актуальные инструменты сборки.

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

??? question "Конфликты версий TensorFlow/PyTorch"
    Установите сборки фреймворков, соответствующие вашей платформе (CUDA/CPU/macOS), затем переустановите `tensoraerospace`.

??? question "macOS (M‑series)"
    Используйте Python 3.10–3.12. При необходимости переустановите `tensorflow-macos` совместимой версии.

??? question "Проблемы с правами/сетью"
    Попробуйте чистое виртуальное окружение или запуск в Docker. Для корпоративных прокси настройте переменные окружения `PIP_*`.

## Следующие шаги

[:material-play-circle-outline: Examples](../example/enviroment/gymnasium.md){ .md-button .md-button--primary }
[:material-airplane-takeoff: Models](../model/f16.md){ .md-button }
[:material-robot-outline: Algorithms](../agent/sac.md){ .md-button }
[:material-book-open-variant: Tutorials](../lesson/0intro.md){ .md-button }
