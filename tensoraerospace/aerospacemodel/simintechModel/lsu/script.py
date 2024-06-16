import subprocess
import numpy as np
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step


## Запускает SimInTech с указанными параметрами
def process():

    PATH_SIT = r'D:\SimInTech64\bin\mmain'
    PATH_PRJ_FILE = r'.\lsu2.xprt'

    # Запускаем SimInTech с проектом в текущей декриктории
    p = subprocess.run(
        f'{PATH_SIT} {PATH_PRJ_FILE} /run /exitonstop',
    )
    p.check_returncode()


if __name__== "__main__":

    dt = 0.01  # Дискретизация
    tp = generate_time_period(tn=100, dt=dt) # Временной периуд
    tps = convert_tp_to_sec_tp(tp, dt=dt)
    number_time_steps = len(tp) # Количество временных шагов
    reference_signals = np.reshape(unit_step(degree=1, tp=tp, time_step=0.5, output_rad=True), [1, -1]) # Заданный сигнал

    f = open('sit_in_1.dat','w')  
    f.write(reference_signals)  
    f.close()

    process()
    