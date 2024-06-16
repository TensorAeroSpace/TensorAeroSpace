Интеграция объекта управления из ПО SimInTech
======================================

Интеграция осуществляется с помощью скрипта, написанным на ЯП Python

.. code-block::
    
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
        

Подключение ОУ из ПО SimInTech в язык программирования Python осуществляется при помощи встроенной библиотеки ЯП Python subprocess, позволяющая создавать процессы запуска программ. Файлы, созданные в ПО SimInTech, можно запускать как с помощью двойного щелчка левой кнопки мыши по ярлыку программы, так и при помощи командной строки Windows PowerShell, написав в нее команду “{расположение папки установки SimInTech}\bin\mmain {расположение интересующего файла в формате prt/xprt}”. Например, “D:\SimInTech64\bin\mmain C:\tensoraerospace\aerospacemodel\simintechModel\lsu2.xprt”. Cтрока графического редактора 10 отражает расположение файла ‘mmain’, строка 12 – расположение файла SimInTech. В строке 14 вызывается модуль subprocess для создания процесса выполнения расчета в ПО SimInTech, с дополнительными параметрами /run и /exitonstop. Первый параметр позволяет запустить наш расчет, второй – закрывает ПО при успешном выполнении операции. На строках 22-26 создается входной ступенчатый сигнал из библиотеки TensorAeroSpace с амплитудой в 1 градус и началом активации 0.5 секунд с шагом дискретизации 0.01 секунда. На строках 28-30 создается файл “sit_in_1.dat” , в который сохраняются значения входного сигнала.