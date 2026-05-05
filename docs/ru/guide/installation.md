# Установка

> :material-rocket-launch-outline: Запустите TensorAeroSpace за 60 секунд — от `pip install` до сходящегося trim-решения.

!!! tip "TL;DR"
    ```bash
    pip install tensoraerospace
    python -c "import tensoraerospace; from tensoraerospace.aerospacemodel.b747.nonlinear import trim; r = trim(altitude_ft=20_000.0, V_ft_s=674.0); print(f'B-747 trim residual = {r.residual:.2e}, converged = {r.converged}')"
    # → B-747 trim residual = 8.66e-14, converged = True
    ```

---

## Системные требования

| Компонент | Минимум | Рекомендовано |
|---|---|---|
| **ОС** | Linux x86_64, Windows 10, macOS 13 | Ubuntu 22.04 LTS / Windows 11 / macOS 14 |
| **Python** | 3.10 | 3.11 или 3.12 |
| **CPU** | 4 ядра, AVX | 8+ ядер, AVX2 / FMA |
| **RAM** | 8 ГБ | 16–32 ГБ для RL-обучения |
| **Диск** | 2 ГБ | 5 ГБ (PyTorch wheels + checkpoints) |
| **GPU** (опц.) | — | NVIDIA RTX с ≥ 8 ГБ VRAM, CUDA 12.2 |
| **MATLAB** (опц.) | — | R2022b+ для Simulink-bridge примера |
| **Unity** (опц.) | — | 2021.3.5f1 / 2023.2.20f1 для Unity ML-Agents bridge |

**Почему такие ограничения?** PyTorch wheels на PyPI содержат скомпилированные CUDA-ядра и требуют AVX. Опубликованные нелинейные модели (B-747, X-15, B-737) интегрируются с шагом `dt = 0.01 с` и RK4 — один 60-секундный эпизод считается меньше секунды на современном ноутбуке, поэтому мощное железо нужно только для RL-обучения, а не для синтеза управления.

| :material-python: Python | Статус | Комментарий |
|---:|:---:|---|
| 3.10 | :material-check-circle: | Минимум |
| 3.11 | :material-check-circle: | **Рекомендуется** |
| 3.12 | :material-check-circle: | Рекомендуется для свежего PyTorch |
| 3.13 | :material-flask-outline: | Экспериментально — некоторые опц. зависимости отстают |
| ≤ 3.9 | :material-close-circle: | Не поддерживается (используется синтаксис 3.10+) |

---

## Быстрая установка (PyPI)

=== "pip"

    ```bash
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

=== "poetry"

    Рекомендуется для проектов с lockfile-фиксацией зависимостей:

    ```bash
    poetry add tensoraerospace
    ```

=== "conda"

    ```bash
    conda create -n tas python=3.11 -y
    conda activate tas
    pip install -U pip setuptools wheel
    pip install tensoraerospace
    ```

=== "uv"

    Быстрая современная альтернатива pip:

    ```bash
    uv venv --python 3.11
    source .venv/bin/activate     # Windows: .venv\Scripts\activate
    uv pip install tensoraerospace
    ```

!!! info "Размер wheel"
    Сам wheel небольшой (~ 5 МБ), но тянет PyTorch (~ 800 МБ), Gymnasium, NumPy, SciPy, matplotlib. Первая установка занимает 1–3 минуты в зависимости от пропускной способности.

---

## Проверка установки

Запустите три проверки ниже — они покрывают (1) Python-импорты, (2) регистрацию Gymnasium-сред, (3) численную корректность trim-солвера.

### 1. Импорт модуля

```bash
python -c "import tensoraerospace as tas; print('TensorAeroSpace', tas.__version__, 'OK')"
```

Ожидаемый вывод: `TensorAeroSpace 0.3.x OK` (без traceback).

### 2. Регистрация сред

```python
import gymnasium as gym
import tensoraerospace  # регистрирует ~ 30 сред

# Все нелинейные 6-DoF самолётные среды:
for env_id in [
    "NonlinearLongitudinalF16-v0",
    "NonlinearAngularF16-v0",
    "NonlinearB747-v0",
    "NonlinearB737-v0",
    "NonlinearX15-v0",
    "NonlinearSkywalkerX8-v0",
    "NonlinearAAIShadow-v0",
]:
    env = gym.make(env_id, trim_at=(20_000.0, 674.0)
                   if env_id == "NonlinearB747-v0" else None,
                   number_time_steps=10)
    print(f"  ✓ {env_id}")
```

### 3. Численная корректность (сходимость trim)

```python
from tensoraerospace.aerospacemodel.b747.nonlinear import trim

result = trim(altitude_ft=20_000.0, V_ft_s=674.0)
assert result.converged, "trim solver должен сойтись"
assert result.residual < 1e-6, f"residual слишком большой: {result.residual}"
print(f"B-747 trim @ FL200, V=674 ft/s: residual = {result.residual:.2e} ✓")
# → B-747 trim @ FL200, V=674 ft/s: residual = 8.66e-14 ✓
```

Если все три прошли — установка **полностью работает**: и Python-обвязка, и численные вычисления корректны.

### Опционально: полный test-сет (только при dev-установке)

```bash
poetry run pytest tests/aerospacemodel/ tests/envs/ -q
# → 894 passed in ~ 19 s
```

---

## Ускорение на GPU

=== "NVIDIA CUDA"

    PyTorch wheels на PyPI по умолчанию идут с CUDA 12.x. Для проверки:

    ```python
    import torch
    print("CUDA доступна:", torch.cuda.is_available())
    print("Кол-во устройств:", torch.cuda.device_count())
    print("Версия CUDA:", torch.version.cuda)
    ```

    Для конкретных версий CUDA сначала установите соответствующий PyTorch:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install tensoraerospace
    ```

    См. [официальную PyTorch install matrix](https://pytorch.org/get-started/locally/).

=== "Apple Silicon (M1/M2/M3)"

    Нативный backend Metal Performance Shaders (MPS):

    ```python
    import torch
    print("MPS доступен:", torch.backends.mps.is_available())
    # → MPS доступен: True
    ```

    Pretrained-агенты автоматически определяют MPS:

    ```python
    from tensoraerospace.agent.sac import SAC
    agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
    # device выбирается автоматически: "mps" на Apple Silicon
    ```

=== "Только CPU"

    Нет GPU? Default-wheels работают:

    - **Синтез управления** (PID, MPC, классическое ADP, IHDP) — CPU достаточно.
    - **Deep RL обучение** (SAC, PPO, DDPG) — медленнее, но возможно (в 10–50 раз медленнее GPU).
    - **Trim / симуляция** — только CPU, без выгоды от GPU.

    Чтобы установить CPU-only PyTorch (меньший размер):

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install tensoraerospace
    ```

---

## Опциональные зависимости

| Возможность | Команда установки | Когда нужно |
|---|---|---|
| **Hugging Face Hub** интеграция | bundled — уже установлено | Загрузка pretrained-агентов (`from_pretrained`), публикация моделей |
| **Unity ML-Agents** | `pip install mlagents-envs==0.30.0` | Запуск [Unity airplane env](../guide/unity_env.md) |
| **MATLAB / Simulink bridge** | MATLAB R2022b+ + `python -m matlab.engine.install` | Запуск [Simulink-примеров](../example/simulink/sim_pyth.md) |
| **3D flight viewer** | bundled (использует three.js) | `env.render()` возвращает интерактивную 3D-сцену |
| **Optuna hyperparam search** | `pip install optuna` | [Cookbook по оптимизации гиперпараметров](../cookbook/07_optuna_search.md) |
| **TensorBoard logging** | `pip install tensorboard` | Метрики в реальном времени при обучении агентов |

---

## Установка из исходников (разработка)

Когда нужно **модифицировать библиотеку**, запустить тестовый набор или собрать документацию локально:

=== "Poetry (рекомендуется)"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace

    poetry install --with dev   # main + dev зависимости (pytest, mkdocs и т.д.)
    poetry shell                 # активация venv

    # Запуск тестов
    poetry run pytest tests/aerospacemodel/ tests/envs/ -q

    # Локальная сборка docs
    poetry run mkdocs serve -a 0.0.0.0:8000
    ```

=== "pip + venv"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace

    python -m venv .venv
    source .venv/bin/activate    # Windows: .venv\Scripts\activate

    pip install -U pip setuptools wheel
    pip install -e ".[dev]"

    pytest tests/ -q
    ```

После клонирования запустите быстрый smoke-тест:

```bash
poetry run python example/aircraft/example_b747_nonlinear.py
# → Trim @ FL200, V=674 ft/s: alpha=+3.603°, delta_e=-0.722°, throttle=0.555
# → Healthy step response, damaged step response, trajectory plot saved
```

---

## Запуск через Docker

!!! info "Рекомендовано для воспроизводимых окружений"
    Официальный образ содержит JupyterLab и все 101 example-notebook готовыми к запуску. Никакой Python-настройки на хосте не требуется.

=== "Pull опубликованного образа"

    ```bash
    docker pull ghcr.io/tensoraerospace/tensoraerospace:latest

    docker run --rm -it -p 8888:8888 \
      -v "$(pwd)/projects:/workspace/projects" \
      ghcr.io/tensoraerospace/tensoraerospace:latest
    ```

    Откройте URL, выведенный в терминале (обычно `http://127.0.0.1:8888`), и перейдите к `/workspace/example/quickstart.ipynb`.

=== "Локальная сборка"

    ```bash
    git clone https://github.com/TensorAeroSpace/TensorAeroSpace.git
    cd TensorAeroSpace
    docker build -t tas:local . --platform=linux/amd64

    docker run --rm -it -p 8888:8888 \
      -v "$(pwd)/projects:/workspace/projects" \
      tas:local
    ```

=== "Проброс GPU"

    Требует [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

    ```bash
    docker run --rm -it --gpus all -p 8888:8888 \
      -v "$(pwd)/projects:/workspace/projects" \
      ghcr.io/tensoraerospace/tensoraerospace:latest
    ```

---

## Обновление и удаление

=== "Обновить"

    ```bash
    # pip
    pip install -U tensoraerospace

    # poetry
    poetry update tensoraerospace
    ```

    Чтобы получить последние нерелизные изменения из `main`:

    ```bash
    pip install -U git+https://github.com/TensorAeroSpace/TensorAeroSpace.git@main
    ```

=== "Удалить"

    ```bash
    pip uninstall tensoraerospace
    # или
    poetry remove tensoraerospace
    ```

    Чтобы освободить место от PyTorch и других зависимостей:

    ```bash
    pip uninstall tensoraerospace torch numpy scipy gymnasium matplotlib
    ```

---

## Решение проблем

???+ question "ImportError: No module named tensoraerospace"
    Пакет не виден активному Python. Проверьте:

    ```bash
    which python && python -c "import sys; print(sys.executable)"
    pip show tensoraerospace
    ```

    Оба должны указывать на одно и то же окружение. Если нет — активируйте правильный venv (`source .venv/bin/activate`, `poetry shell` или `conda activate tas`).

??? question "Конфликты версий PyTorch (`undefined symbol`, `RuntimeError: CUDA error`)"
    Скорее всего, версия CUDA в PyTorch не совпадает с системным драйвером, или используется устаревший wheel-кэш.

    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip cache purge
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install --upgrade --force-reinstall tensoraerospace
    ```

??? question "macOS (Apple Silicon) — torch не находит OpenMP"
    Apple Xcode CLT поставляется без OpenMP. Либо установите через brew (`brew install libomp`), либо используйте встроенный MPS-backend PyTorch (для inference OpenMP не нужен).

??? question "Trim solver не сходится для моей точки (h, V)"
    Для большинства самолётов это значит, что точка находится за пределами cruise-envelope. Проверьте:

    - **Ниже стола**: `V` слишком мал для требуемой подъёмной силы при текущем весе.
    - **Выше потолка**: плотность слишком мала для тяги двигателя.
    - **X-15** в частности: ракетный двигатель не имеет уровневого крейса — используйте `gamma_rad=…` для подъёма или `level_trim()` для нефизичного случая. См. [trim в X-15](../model/x15_nonlinear.md#об-отсутствии-trim-конверта).

??? question "Permissions / корпоративный proxy"
    Используйте `pip install --user` (без админских прав) или Docker. Для корпоративного proxy задайте `PIP_INDEX_URL` / `HTTP_PROXY`:

    ```bash
    export PIP_INDEX_URL=https://your-mirror.example.com/simple
    export HTTP_PROXY=http://proxy:8080
    pip install tensoraerospace
    ```

??? question "CUDA доступна, но агент работает на CPU"
    У некоторых pretrained-агентов device захардкожен в чекпойнте. Принудительно переместите:

    ```python
    import torch
    agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.policy = agent.policy.to(device)
    agent.critic = agent.critic.to(device)
    agent.critic_target = agent.critic_target.to(device)
    agent.device = device
    ```

??? question "`Could not find platform-independent libraries` на Windows"
    Обычно повреждённая установка Python. Переустановите Python 3.11 с [python.org](https://www.python.org/downloads/) (не используйте Windows Store-версию), затем пересоздайте venv.

---

## Что дальше?

Готовы строить.

[:material-rocket: 30-секундный quickstart](../cookbook/01_hello.md){ .md-button .md-button--primary }
[:material-airplane-takeoff: Модели](../model/f16.md){ .md-button }
[:material-robot-outline: Алгоритмы](../agent/sac.md){ .md-button }
[:material-book-open-variant: 16-рецептовый cookbook](../cookbook/01_hello.md){ .md-button }
[:material-school-outline: Уроки](../lesson/base/tutor_1.md){ .md-button }

Если столкнулись с проблемой, не покрытой troubleshooting-секцией — пишите в [GitHub Discussions](https://github.com/TensorAeroSpace/TensorAeroSpace/discussions) или открывайте [issue](https://github.com/TensorAeroSpace/TensorAeroSpace/issues).
