Интегрирование Simulink модели в Python 
=======================================

Интегрирование Simulink модели в Python осуществлялось с помощью DLL библиотеки (библиотеки динамической компоновки). В результате генерации cpp кода получили код Simulink модели и Makefile с расширением .mk. После его запуска была скомпилирована DLL библиотека.

Для согласования типов данных и корректной работы модели использовался модуль ctypes и был написан преобразователь типов. tensorairspace/aircraftmodel/model/rtwtypes.py

.. code-block:: python

    import os
    import ctypes
    import matplotlib.pyplot as plt
    from tensorairspace.aircraftmodel.model.rtwtypes import *

    # загружаем dll файл
    b747_dll_path = os.path.abspath("../tensorairspace/aircraftmodel/model/simulinkModel/b747/b747_model_win64.dll")
    b747_dll = ctypes.windll.LoadLibrary(b747_dll_path)

    b747_model_initialize = b747_dll.b747_model_initialize
    b747_model_step = b747_dll.b747_model_step # шаг модели определяется в slx файле
    b747_model_terminate = b747_dll.b747_model_terminate

    # входные параметры
    ref_signal = real32_T.in_dll(b747_dll, "b747_model_U")

    # выходные параметры
    b747_Y = ExtY_T.in_dll(b747_dll, "b747_model_Y")

    b747_model_initialize()

    b747_time = []
    b747_u = []
    b747_w = []
    b747_q = []
    b747_theta = []

    for step in range(int(2100)):
        b747_model_step()
        
        b747_time.append(float(b747_Y.time))
        b747_u.append(float(b747_Y.u))
        b747_w.append(float(b747_Y.w))
        b747_q.append(float(b747_Y.q))
        b747_theta.append(float(b747_Y.theta))

    b747_model_terminate()

Пример использования Simulink модели с Python хранится в example/example_sim_model_to_python.ipynb