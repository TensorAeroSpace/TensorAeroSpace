# SAC для управления тангажом Boeing 747

<div class="admonition tip">
<p class="admonition-title">Что вы узнаете</p>
<p>Это руководство демонстрирует, как оценить предобученного агента <strong>Soft Actor-Critic (SAC)</strong> для продольного управления тангажом самолёта Boeing 747 с использованием нормализованной среды <code>ImprovedB747Env</code>.</p>
</div>

![b747](img/sac-b747-impoved.jpg)

---

## Обзор

Алгоритм Soft Actor-Critic (SAC) -- это современный метод глубокого обучения с подкреплением off-policy, отлично подходящий для задач непрерывного управления. Этот пример демонстрирует:

- **Предобученный агент**: загрузка готовой политики SAC из Hugging Face Hub
- **Динамика Boeing 747**: реалистичная модель продольной динамики полёта
- **Нормализованная среда**: действия и наблюдения масштабированы в [-1, 1] для стабильного обучения
- **Визуализация в реальном времени**: рендеринг реакции самолёта на основе Pygame

### Описание задачи

Агент управляет **отклонением руля высоты** для отслеживания синусоидального эталонного сигнала угла тангажа. Состояние включает:

| Переменная | Описание | Единица |
|------------|----------|---------|
| `u` | Возмущение продольной скорости | м/с |
| `w` | Возмущение вертикальной скорости | м/с |
| `q` | Угловая скорость тангажа | рад/с |
| `theta` | Угол тангажа | рад |

---

## Установка

### Быстрая установка

```bash
pip install -U tensoraerospace pygame torch
```

### Зависимости

| Пакет | Назначение | Версия |
|-------|-----------|--------|
| `tensoraerospace` | Основная библиотека со средами и агентами | Последняя |
| `pygame` | Визуализация в реальном времени | >=2.0.0 |
| `torch` | Бэкенд нейронных сетей для SAC | >=1.9.0 |

<div class="admonition note">
<p class="admonition-title">Требуется дисплей</p>
<p>Рендеринг использует Pygame и требует графический дисплей. Для серверов без дисплея удалите вызов <code>env.render()</code> или используйте виртуальный дисплей (например, <code>xvfb</code>).</p>
</div>

---

## Быстрый старт

### Запуск из командной строки

Запустите предобученного агента с параметрами по умолчанию (автоопределение GPU):

```bash
python example/reinforcement_learning/sac-b747-render.py \
    --render \
    --dt 0.1 \
    --tn 200 \
    --repo TensorAeroSpace/sac-b747
```

Или явно укажите устройство:

```bash
python example/reinforcement_learning/sac-b747-render.py \
    --render \
    --dt 0.1 \
    --tn 200 \
    --repo TensorAeroSpace/sac-b747 \
    --device cuda  # Используйте GPU (или 'mps' для Apple Silicon, 'cpu' для CPU)
```

#### Аргументы командной строки

| Аргумент | Описание | По умолчанию |
|----------|----------|--------------|
| `--render` | Включить визуализацию в реальном времени | `False` |
| `--dt` | Шаг моделирования (секунды) | `0.1` |
| `--tn` | Количество временных шагов | `200` |
| `--repo` | Репозиторий Hugging Face Hub | `TensorAeroSpace/sac-b747` |
| `--device` | Устройство вычислений (`cuda`, `mps`, `cpu`) | Автоопределение |
| `--seed` | Случайное зерно для воспроизводимости | `42` |

---

## Полный пример на Python

### Шаг 1: Импорт зависимостей

```python
import numpy as np
from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
```

### Шаг 2: Настройка параметров моделирования

```python
# Настройки моделирования
dt = 0.1    # Шаг по времени в секундах (частота обновления 10 Гц)
tn = 200    # Количество шагов (20 секунд всего)
```

### Шаг 3: Генерация эталонного сигнала

Создание плавного синусоидального эталонного угла тангажа с амплитудой 1 градус:

```python
# Генерация временных массивов
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)

# Создание эталонного сигнала: синусоида 1 градус на частоте 0.05 Гц
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps),
        frequency=0.05,          # Период 20 секунд
        amplitude=np.deg2rad(1.0),  # Преобразование 1 градуса в радианы
        vertical_shift=0.0       # Центрирование вокруг 0 градусов
    ),
    (1, -1),  # Изменение формы в (1, tn)
)
```

<div class="admonition info">
<p class="admonition-title">Параметры сигнала</p>
<p>Эталонный сигнал имеет период <code>1/0.05 = 20 секунд</code>, то есть самолёт совершает ровно один цикл колебаний за эпизод.</p>
</div>

### Шаг 4: Инициализация среды

```python
# Определение начального состояния: [u, w, q, theta] -- все нули (балансировочный режим)
initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)

# Создание улучшенной среды B747
env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=len(tp),
    dt=dt,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
)

# Синхронизация дискретизации модели с шагом среды
env.unwrapped.model.discretisation_time = dt
```

#### Конфигурация среды

| Параметр | Значение | Описание |
|----------|---------|----------|
| `initial_state` | `[0, 0, 0, 0]` | Балансировочный режим полёта |
| `dt` | `0.1` | Дискретный шаг по времени |
| `initial_elevator_deg` | `0.0` | Нейтральное положение руля высоты |
| `use_initial_action_on_first_step` | `True` | Применить начальное действие немедленно |

### Шаг 5: Загрузка предобученного агента

```python
import torch

# Автоопределение устройства (CUDA/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else 
                     ("mps" if hasattr(torch.backends, "mps") and 
                      torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Скачивание и загрузка предобученного агента SAC из Hugging Face Hub
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")

# Перенос агента на выбранное устройство (если отличается от сохранённого)
if agent.device != device:
    print(f"Moving agent from {agent.device} to {device}")
    agent.device = device
    agent.critic = agent.critic.to(device)
    agent.critic_target = agent.critic_target.to(device)
    agent.policy = agent.policy.to(device)
    # Перенос log_alpha, если существует (для автоматической настройки энтропии)
    if hasattr(agent, "log_alpha") and agent.log_alpha is not None:
        agent.log_alpha = agent.log_alpha.to(device)
```

<div class="admonition success">
<p class="admonition-title">Интеграция с Hugging Face</p>
<p>Модель автоматически скачивается из Hub при первом использовании и кэшируется локально. Ручное скачивание не требуется!</p>
</div>

<div class="admonition tip">
<p class="admonition-title">Поддержка GPU</p>
<p>Скрипт автоматически определяет и использует GPU (CUDA/MPS), если доступен. Вы также можете явно указать устройство с помощью аргумента командной строки <code>--device</code>. Агент будет автоматически перенесён на выбранное устройство после загрузки.</p>
</div>

### Шаг 6: Запуск цикла оценки

```python
# Сброс среды и получение начального наблюдения
obs, info = env.reset()
done = False
ret = 0.0  # Суммарная награда

# Цикл эпизода
while not done:
    # Получение детерминированного действия от агента (без исследования)
    action = agent.select_action(obs, evaluate=True)
    
    # Шаг среды
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Рендеринг визуализации (закомментируйте для безголового режима)
    env.render(mode="human")
    
    # Проверка завершения
    done = bool(terminated or truncated)
    ret += float(reward)

# Вывод итоговой производительности
print(f"Episode Return: {ret:.2f}")
```

### Ожидаемый результат

```
Episode Return: 1847.32
```

<div class="admonition tip">
<p class="admonition-title">Интерпретация результатов</p>
<p>Более высокая суммарная награда означает лучшее отслеживание эталонного сигнала. Хорошо обученный агент обычно достигает суммарной награды выше 1500 для этой задачи.</p>
</div>

---

## Понимание результатов

### На что обратить внимание

При запуске с `env.render()` вы увидите:

1. **Состояние самолёта**: графики скорости, угловой скорости тангажа и угла тангажа в реальном времени
2. **Управляющее воздействие**: отклонение руля высоты во времени
3. **Отслеживание эталона**: насколько точно угол тангажа следует за синусоидой
4. **Сигнал награды**: мгновенная награда на каждом шаге

### Метрики производительности

Успешный агент демонстрирует:

- **Малая ошибка отслеживания**: угол тангажа точно следует за эталоном
- **Плавное управление**: отклонения руля высоты без чрезмерных колебаний
- **Стабильная динамика**: отсутствие расходимости или нестабильности
- **Высокая суммарная награда**: обычно > 1500

---

## Ключевые концепции

### Нормализация в ImprovedB747Env

<div class="admonition warning">
<p class="admonition-title">Важно</p>
<p>Все действия и наблюдения <strong>нормализованы в диапазоне [-1, 1]</strong>. Среда выполняет масштабирование внутренне:</p>
<ul>
<li><strong>Действия</strong>: выходы сети [-1, 1] -> отображаются в физические пределы руля высоты</li>
<li><strong>Наблюдения</strong>: физические состояния -> нормализуются в [-1, 1] для входа нейронной сети</li>
</ul>
</div>

### Особенности алгоритма SAC

**Soft Actor-Critic** объединяет:

- **RL с максимальной энтропией**: стимулирование исследования через регуляризацию энтропии
- **Off-policy обучение**: эффективное использование данных, обучение из буфера воспроизведения
- **Архитектура Актёр-Критик**: раздельные сети политики и ценности
- **Автоматическая настройка температуры**: адаптивный баланс исследования и эксплуатации

Подробнее: [Документация SAC](../../../agent/sac.md)

### Синхронизация времени

```python
env.unwrapped.model.discretisation_time = dt
```

Эта строка **критически важна** для обеспечения того, чтобы модель непрерывной динамики использовала ту же дискретизацию, что и шаг среды. Несоответствие может привести к:

- Нестабильности моделирования
- Плохой производительности агента
- Некорректному расчёту наград

---

## Устранение неполадок

### Частые проблемы

<details>
<summary><strong>ImportError: No module named 'pygame'</strong></summary>

**Решение**: Установите pygame для поддержки визуализации:
```bash
pip install pygame
```

Для сред без дисплея удалите вызов `env.render()`.
</details>

<details>
<summary><strong>Ошибка загрузки модели или превышение времени ожидания</strong></summary>

**Решение**: Проверьте интернет-соединение и статус Hugging Face Hub. Вы также можете скачать вручную:
```python
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747", access_token="your_token")
```
</details>

<details>
<summary><strong>Низкая производительность / плохое отслеживание</strong></summary>

**Решение**: Убедитесь, что:
1. Дискретизация модели совпадает с `dt`: `env.unwrapped.model.discretisation_time = dt`
2. Амплитуда эталонного сигнала разумна (1-5 градусов)
3. Используется `evaluate=True` для детерминированных действий
</details>

<details>
<summary><strong>Ошибка дисплея Pygame на удалённом сервере</strong></summary>

**Решение**: Используйте виртуальный дисплей или отключите рендеринг:
```bash
# С виртуальным дисплеем
xvfb-run -a python your_script.py

# Или закомментируйте в коде
# env.render(mode="human")
```
</details>

<details>
<summary><strong>GPU не используется / Модель работает на CPU</strong></summary>

**Решение**: Скрипт автоматически определяет GPU, но вы можете указать явно:
```bash
# Явное использование CUDA
python example/reinforcement_learning/sac-b747-render.py --device cuda

# Использование MPS (Apple Silicon)
python example/reinforcement_learning/sac-b747-render.py --device mps

# Принудительно CPU
python example/reinforcement_learning/sac-b747-render.py --device cpu
```

Скрипт автоматически перенесёт загруженную модель на указанное устройство. Проверьте вывод консоли для информации об устройстве:
```
Using device: cuda
CUDA device: NVIDIA GeForce RTX 3090
CUDA memory: 24.00 GB
Agent device: cuda
Policy device: cuda
Critic device: cuda
```
</details>

<details>
<summary><strong>Segmentation fault или чёрный экран</strong></summary>

**Решение**: Обычно это указывает на проблемы рендеринга в безголовых системах:
1. **Отключите рендеринг**: Используйте флаг `--no-render`
2. **Проверьте DISPLAY**: Убедитесь, что переменная окружения `DISPLAY` установлена для X11
3. **Используйте виртуальный дисплей**: `xvfb-run -a python sac-b747-render.py`
4. **Проверьте GPU**: Убедитесь, что драйверы GPU правильно установлены при использовании CUDA

Скрипт теперь включает автоматическое определение GPU и управление устройствами для предотвращения таких проблем.
</details>

---

## Связанные примеры

- [Управление истребителем F-16 с SAC](example-sac-f16.md)
- [Документация алгоритма SAC](../../../agent/sac.md)
- [Руководство по установке](../../../guide/installation.md)
- [Интеграция среды Unity](../../../guide/unity_env.md)

---

## Дополнительные ресурсы

- [Предобученная модель на Hugging Face](https://huggingface.co/TensorAeroSpace/sac-b747)
- [Организация TensorAeroSpace](https://huggingface.co/TensorAeroSpace)
- [Статья SAC](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)
- [Документация модели Boeing 747](../../../model/b747.md)

---

<div class="admonition question">
<p class="admonition-title">Нужна помощь?</p>
<p>
Присоединяйтесь к нашему сообществу:
</p>
<ul>
<li><a href="https://github.com/TensorAeroSpace/TensorAeroSpace/discussions">GitHub Discussions</a></li>
<li><a href="https://github.com/TensorAeroSpace/TensorAeroSpace/issues">Сообщить о проблеме</a></li>
<li><a href="https://github.com/TensorAeroSpace/TensorAeroSpace">Star на GitHub</a></li>
</ul>
</div>
