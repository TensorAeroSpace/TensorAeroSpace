import subprocess

import numpy as np

from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period


## Запускает SimInTech с указанными параметрами
def process():
    """
    Запускает SimInTech с указанными параметрами проекта

    Функция выполняет запуск SimInTech с проектом lsu2.xprt в режиме выполнения
    с автоматическим завершением после остановки симуляции.

    Raises:
        subprocess.CalledProcessError: Если процесс SimInTech завершился с ошибкой

    Note:
        Требует наличия SimInTech в указанном пути и файла проекта lsu2.xprt
        в текущей директории
    """
    PATH_SIT = r"D:\SimInTech64\bin\mmain"
    PATH_PRJ_FILE = r".\lsu2.xprt"

    # Запускаем SimInTech с проектом в текущей декриктории
    p = subprocess.run(
        f"{PATH_SIT} {PATH_PRJ_FILE} /run /exitonstop",
    )
    p.check_returncode()


if __name__ == "__main__":
    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=100, dt=dt)  # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp)  # Количество временных шагов
    reference_signals = np.reshape(
        unit_step(degree=1, tp=tp, time_step=0.5, output_rad=True), [1, -1]
    )  # Заданный сигнал

    f = open("sit_in_1.dat", "w")
    f.write(reference_signals)
    f.close()

    process()
