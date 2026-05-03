# Генерация dll/so библиотеки

## Генерация C++ кода

> Требования: Embedded Coder, корректно настроенный компилятор (GCC/Visual Studio), доступ к сгенерированным заголовкам `MODEL_NAME.h`.

Для поддержки ОУ из Simulink необходим Embedded Coder.

Шаги:

1. Описать входы/выходы блоками In1/Out1.
2. В настройках: Code Generation → System target file = `ert_shrlib.tlc`.

![Настройка Codegen](img/image019.png){ width=400 }

1. Собрать модель (Ctrl+B или Build model). Появится каталог с исходниками C/C++.

## Интеграция Simulink‑модели в Python

### 1) Сборка динамической библиотеки

— Linux/macOS:

```bash
gcc -shared -o model.so -fPIC *.c
```

— Windows (пример):

```bash
make -f MODEL_NAME.mk
```

Где `*.c` — все сгенерированные исходники.

### 2) Описание интерфейса взаимодействия

Интерфейс описывается через `ctypes.Structure` и типы из `tensoraerospace/aerospacemodel/utils/rtwtypes.py`.

```python
import ctypes
from tensoraerospace.aerospacemodel.utils.rtwtypes import real_T

class ExtY(ctypes.Structure):
    _fields_ = [
        ("u", real_T),
        ("w", real_T),
        ("q", real_T),
        ("theta", real_T),
        ("sim_time", real_T),
    ]

class ExtU(ctypes.Structure):
    _fields_ = [("ref_signal", real_T)]
```

В библиотеке обычно доступны функции:

- `MODEL_NAME_initialize`
- `MODEL_NAME_step` (шаг равен `dt` из параметров модели)
- `MODEL_NAME_terminate`

### 3) Пример использования из Python

В этом примере `ExtY` объявлен **без** поля `sim_time` — упрощённая
4-полевая версия для случая, когда сборка Embedded Coder не выводит время
симуляции наружу. Если `sim_time` присутствует в сгенерированной структуре
(см. п. 2 выше), используйте 5-полевой вариант — порядок полей в `_fields_`
обязан **точно совпадать** с заголовком `MODEL_NAME.h`, иначе чтение через
`in_dll(...)` вернёт мусор. Какой вариант брать — определяется тем, как
вы сконфигурировали порты модели в Simulink.

```python
import os
import ctypes
import matplotlib.pyplot as plt
from tensoraerospace.aerospacemodel.utils.rtwtypes import real_T

class ExtY(ctypes.Structure):
    # 4-полевой вариант: sim_time в эту сборку не выведен.
    # Если в MODEL_NAME.h структура содержит sim_time, добавьте поле сюда.
    _fields_ = [("u", real_T), ("w", real_T), ("q", real_T), ("theta", real_T)]

class ExtU(ctypes.Structure):
    _fields_ = [("ref_signal", real_T)]

# Укажите корректное имя/расширение библиотеки
lib_path = os.path.abspath("model.dll")  # или model.so / model.dylib
dll = ctypes.CDLL(lib_path)

X = ExtU.in_dll(dll, 'uav1_model_U')
Y = ExtY.in_dll(dll, 'uav1_model_Y')

model_initialize = dll.model_initialize
model_step = dll.model_step
model_terminate = dll.model_terminate

model_initialize()

u_vals, w_vals, q_vals, theta_vals = [], [], [], []
for _ in range(2100):
    X.ref_signal = -0.0
    model_step()
    u_vals.append(Y.u)
    w_vals.append(Y.w)
    q_vals.append(Y.q)
    theta_vals.append(Y.theta)

model_terminate()

plt.plot(w_vals)
plt.ylabel('$u$, [м/с]')
plt.show()
```
