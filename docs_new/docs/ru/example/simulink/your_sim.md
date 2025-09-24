Интеграция собственной Simulink‑модели
======================================

Ниже — короткий и практичный путь: сгенерировать код из Simulink, собрать динамическую библиотеку и запустить вашу модель из Python через `ctypes`.

Цели и требования
------------------

- Что вы получите:
  - **Артефакты сборки** из Simulink (`*.c`, `*.h`) и **динамическую библиотеку** (`.so`/`.dylib`/`.dll`).
  - **Минимальный Python‑скрипт**, который инициализирует модель, делает шаги и читает выходы.
- Что требуется:
  - MATLAB/Simulink с установленным **Embedded Coder**.
  - Компилятор C/C++ (GCC/Clang/MSVC) и утилиты линковки.
  - Python 3.8+ и `matplotlib` для визуализации (опционально).

Шаг 1. Сгенерируйте C/C++ код в Simulink
----------------------------------------

- Добавьте блоки `In1`/`Out1` для описания входов и выходов модели.
- В настройках выберите: `Code Generation → System target file: ert_shrlib.tlc` (Embedded Coder).
- Соберите модель (`Ctrl+B` или кнопка “Build model”). В папке модели появятся `*.c`, `*.h` и прочие артефакты.

![Генерация C/C++ кода](img/cpp_gen.png){ width="400" }

!!! note "Шаг моделирования"
    Шаг расчета задается в параметрах Simulink‑модели. Он будет использован функцией шага в сгенерированном коде.

Структура артефактов и соглашения об именах
-------------------------------------------

После сборки в каталоге модели появятся файлы вида:

- Заголовки: `MODEL_NAME.h`, приватные заголовки `*_private.h`, типы `rtwtypes.h`.
- Исходники: `*.c` с реализацией и точкой входа шага модели.
- Имена глобальных объектов и функций:
  - Входы: `MODEL_NAME_U`
  - Выходы: `MODEL_NAME_Y`
  - Инициализация/шаг/завершение: `MODEL_NAME_initialize`, `MODEL_NAME_step`, `MODEL_NAME_terminate`

Шаг 2. Соберите динамическую библиотеку
--------------------------------------

Сборка выполняется в каталоге, где лежат сгенерированные `*.c` файлы.

=== "Linux"

   ```bash
   gcc -shared -fPIC -O2 -o model.so *.c
   ```

=== "macOS"

   ```bash
   clang -dynamiclib -O2 -o model.dylib *.c
   ```

=== "Windows (MinGW)"

   ```bash
   gcc -shared -O2 -o model.dll *.c
   ```

=== "Windows (MSVC)"

   ```bash
   cl /LD /O2 *.c /Fe:model.dll
   ```

!!! tip "Подсказка"
    Иногда требуется добавить системные библиотеки (например, `-lm` на Linux). Если компоновщик сообщает об отсутствующих символах — проверьте вывод сборки и добавьте необходимые флаги.

Быстрая проверка загрузки библиотеки
------------------------------------

Перед интеграцией в код убедитесь, что библиотека загружается:

```python
import ctypes, os
print(ctypes.CDLL(os.path.abspath("model.so")))  # замените на .dylib или .dll
```

!!! warning "Windows"
    - На Windows может потребоваться, чтобы зависимые DLL находились в одном каталоге или были прописаны в `PATH`.
    - Если видите `OSError: [WinError 126] The specified module could not be found` — проверьте расположение всех зависимостей и архитектуру (x64 vs x86).

Шаг 3. Опишите интерфейс входов/выходов в Python
------------------------------------------------

Откройте сгенерированный заголовок `MODEL_NAME.h` и найдите секции `External inputs` и `External outputs`. По ним создайте структуры для `ctypes`.

```python
import ctypes
from tensoraerospace.aerospacemodel.utils.rtwtypes import real_T


class ExtU(ctypes.Structure):
    """Входы модели (name, type)"""
    _fields_ = [
        ("ref_signal", real_T),
    ]


class ExtY(ctypes.Structure):
    """Выходы модели (name, type)"""
    _fields_ = [
        ("Wz", real_T),
        ("theta_big", real_T),
        ("H", real_T),
        ("alpha", real_T),
        ("theta_small", real_T),
    ]
```

!!! info "Имена и типы"
    Имена полей и их типы всегда берите из `MODEL_NAME.h`. В именах глобальных структур и функций используется имя модели: `MODEL_NAME_U`, `MODEL_NAME_Y`, `MODEL_NAME_initialize`, `MODEL_NAME_step`, `MODEL_NAME_terminate`.

Шаг 4. Пример запуска из Python
--------------------------------

Ниже — минимальный пример, который загружает библиотеку, выполняет 2100 шагов и строит графики. Подставьте актуальное имя модели и путь к библиотеке (`.so`/`.dylib`/`.dll`).

```python
import os
import ctypes
import matplotlib.pyplot as plt

from tensoraerospace.aerospacemodel.utils.rtwtypes import real_T


class ExtU(ctypes.Structure):
    _fields_ = [("ref_signal", real_T)]


class ExtY(ctypes.Structure):
    _fields_ = [
        ("Wz", real_T),
        ("theta_big", real_T),
        ("H", real_T),
        ("alpha", real_T),
        ("theta_small", real_T),
    ]


# Укажите правильный путь и расширение библиотеки вашей модели
lib_path = os.path.abspath("model.so")   # Linux
# lib_path = os.path.abspath("model.dylib")  # macOS
# lib_path = os.path.abspath("model.dll")    # Windows

dll = ctypes.CDLL(lib_path)

# Имя префикса совпадает с именем модели в Simulink (здесь — "model")
X = ExtU.in_dll(dll, "model_U")
Y = ExtY.in_dll(dll, "model_Y")

model_initialize = dll.model_initialize
model_step = dll.model_step
model_terminate = dll.model_terminate

model_initialize()

wz, theta_big, H, alpha, theta_small = [], [], [], [], []

for _ in range(2100):
    X.ref_signal = -0.1
    model_step()
    wz.append(Y.Wz)
    theta_big.append(Y.theta_big)
    H.append(Y.H)
    alpha.append(Y.alpha)
    theta_small.append(Y.theta_small)

model_terminate()

plt.figure(figsize=(8, 6))
plt.subplot(3, 2, 1); plt.plot(wz); plt.title("$w_z$, рад/с")
plt.subplot(3, 2, 2); plt.plot(H); plt.title("H, м")
plt.subplot(3, 2, 3); plt.plot(theta_big); plt.title("$\\Theta$, рад")
plt.subplot(3, 2, 4); plt.plot(theta_small); plt.title("$\\theta$, рад")
plt.subplot(3, 2, 5); plt.plot(alpha); plt.title("$\\alpha$, рад")
plt.tight_layout(); plt.show()
```

Распространенные проблемы и их решения
--------------------------------------

- "Не нахожу `MODEL_NAME_U`/`MODEL_NAME_Y`":
  - Убедитесь, что используете корректный префикс имени модели (как файл `.h`).
  - Проверьте, что `In1`/`Out1` присутствуют и экспортируются.
- Ошибка линковки про `math` или `sqrt`:
  - Добавьте `-lm` при сборке на Linux.
- Python падает при обращении к полям структуры:
  - Проверьте соответствие типов полей `ctypes` тому, что в `MODEL_NAME.h`.
  - Убедитесь, что библиотека собрана для той же архитектуры, что и Python (x64/x86).

Полезные ссылки
---------------

- Репозиторий с примером модели: [tensoraerospace/simulink-example](https://github.com/tensoraerospace/simulink-example)
- Конвертация и запуск Simulink‑моделей: см. также раздел «Simulink to Python» в документации.
