# SimInTech → Python

Интегрируйте модели SimInTech в Python через запуск проекта из скрипта и обмен данными через файлы. Ниже — быстрый старт и рабочий пример.

## Быстрый старт

1. Подготовьте проект SimInTech (`.prt`/`.xprt`) с блоками:
   - чтение входного сигнала из файла (например, `sit_in_1.dat`),
   - запись выходного сигнала в файл (например, `sit_out_1.dat`).
2. Найдите путь к исполняемому файлу SimInTech `mmain.exe` (обычно `C:\\SimInTech64\\bin\\mmain.exe`).
3. Сгенерируйте входной сигнал в Python, сохраните в `sit_in_1.dat`.
4. Запустите проект SimInTech из Python: `mmain.exe <path_to_project> /run /exitonstop`.
5. Прочитайте выходной файл в Python и постройте график.

!!! note "Поддерживаемая платформа"
    Примеры ниже рассчитаны на Windows, так как SimInTech — Windows‑приложение.

---

## Полноценный пример на Python

```python
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standard import unit_step


def run_simintech(sit_bin: Path, project: Path, extra_args: list[str] | None = None, timeout_sec: int | None = 120) -> None:
    """Запускает SimInTech проект с параметрами /run /exitonstop.

    - sit_bin: путь к mmain.exe
    - project: путь к .prt/.xprt
    - extra_args: дополнительные аргументы командной строки
    - timeout_sec: таймаут ожидания завершения, сек
    """
    args = [str(sit_bin), str(project), "/run", "/exitonstop"]
    if extra_args:
        args.extend(extra_args)

    completed = subprocess.run(
        args,
        check=False,           # покажем понятную ошибку ниже
        capture_output=True,   # соберём stdout/stderr на случай ошибок
        text=True,
        timeout=timeout_sec,
        shell=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "SimInTech завершился с ошибкой "
            f"(код {completed.returncode}).\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


if __name__ == "__main__":
    # Пути: скорректируйте под свою установку/проект
    sit_bin = Path(r"C:\\SimInTech64\\bin\\mmain.exe")
    project = Path(r".\\lsu2.xprt")  # проект в текущей папке

    # Генерация входного сигнала: ступенька 1° с t0=0.5 c, dt=0.01 c
    dt = 0.01
    tp = generate_time_period(tn=100, dt=dt)                 # дискретные такты
    tps = convert_tp_to_sec_tp(tp, dt=dt)                     # время в секундах

    # unit_step возвращает массив значений (в радианах при output_rad=True)
    # Сформируем форму [channels, time] — здесь один канал
    ref = unit_step(degree=1, tp=tp, time_step=0.5, output_rad=True)
    reference_signals = np.reshape(ref, (1, -1))

    # Запись входного файла. Часто SimInTech ожидает значения построчно/покомпонентно.
    # Уточните формат своего проекта. В простейшем случае — один столбец значений.
    in_file = Path("sit_in_1.dat")
    np.savetxt(in_file, reference_signals.ravel(), fmt="%.6f")

    # Запуск расчёта
    run_simintech(sit_bin=sit_bin, project=project)

    # Чтение результата (путь и формат выходного файла настройте в проекте SimInTech)
    out_file = Path("sit_out_1.dat")
    if out_file.exists():
        y = np.loadtxt(out_file, dtype=float)

        # Построение графика
        plt.plot(tps, y[: len(tps)])
        plt.xlabel("t, [с]")
        plt.ylabel("y, [ед.]")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"Предупреждение: выходной файл {out_file} не найден. Проверьте настройки проекта SimInTech.")
```

### Пояснения к ключевым строкам

- `run_simintech(...)` использует `subprocess.run` без `shell=True` и с аргументами списком — это надёжнее и безопаснее.
- Флаги `/run` и `/exitonstop` запускают расчёт и закрывают GUI после успешного завершения.
- Вход/выход через `*.dat`: используйте `numpy.savetxt`/`numpy.loadtxt` для простого текстового формата. Для нескольких каналов организуйте столбцы.
- Модули `generate_time_period`, `convert_tp_to_sec_tp` и `unit_step` входят в TensorAeroSpace и упрощают формирование тестовых сигналов.

---

## Альтернатива: запуск из PowerShell

```powershell
"C:\\SimInTech64\\bin\\mmain.exe" \
  "C:\\tensoraerospace\\aerospacemodel\\simintechModel\\lsu2.xprt" \
  /run /exitonstop
```

> Кавычки обязательны, если путь содержит пробелы.

---

## Частые проблемы и решения

- SimInTech не стартует из Python:
  - Проверьте корректность пути к `mmain.exe`.
  - Убедитесь, что проект `.xprt` существует и доступен.
  - Запустите скрипт из «PowerShell (x64)».
- Процесс завершается с ненулевым кодом и пустым результатом:
  - Откройте проект вручную и проверьте блоки ввода/вывода файлов.
  - Сравните формат `sit_in_1.dat` с ожидаемым в проекте (число столбцов, разделитель, кодировка). Для бинарного формата используйте `numpy.tofile`/`fromfile`.
- Выходной файл не создаётся:
  - Убедитесь, что путь записи корректен и каталог существует.
  - Проверьте права доступа к директории.

!!! warning "Формат данных"
    Конкретный формат файлов (`*.dat`) зависит от настроек вашего проекта SimInTech. При несовпадении формата обновите логику записи/чтения в Python.

---

## Чек‑лист

- [ ] Путь к `mmain.exe` верный (или доступен через переменную окружения/ярлык)
- [ ] Проект `.prt/.xprt` открывается вручную и корректно считается
- [ ] Настроены блоки чтения входа и записи выхода в файлы
- [ ] Python генерирует входной файл в нужном формате
- [ ] Запуск через `subprocess.run` завершается без ошибок, выходной файл читается