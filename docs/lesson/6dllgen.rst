Генерация dll/so библиотеки
=========================

Генерация C++ кода
------------------

Для поддержки ОУ из ПО Simulink необходима надстройка для Simulink – Embedded Coder.

Для преобразования Simulink модели в С код:

#. При помощи блоков In1/Out1 опишите входные и выходные параметры

#. 	В настройках Simulink выберите: Code Generation/System target file ert_shrlib.tlc.
	
	.. image:: img/image019.png
  		:width: 400
  		:alt: Блок Stae-Space

#. Для построения модели используйте сочетание клавиш ctrl+B. Также это можно сделать в панели навигации, выбрав пункт “Build model”. В результате появится папка с кодом на языке C++ в директории, в которой находилась модель. 



Интегрирование Simulink модели в Python 
---------------------------------------

#. Создайте so файл

   Интегрирование Simulink модели в Python осуществляется с помощью DLL библиотеки (библиотеки динамической компоновки). Для ее генерации необходим gcc compiler.

   Введите команду

   .. code-block:: 

      gcc -shared -o model.so -fPIC *.c

   где .c - все файлы с расширением c

   В папке появится so файл.

   Для Windows
    .. code-block:: 

        make -f MODEL_NAME.mk

#. Опишите интерфейс взаимодействия

  Интерфейс взаимодействия описывается для входных и выходных параметров при помощи ctypes.Structure и преобразователя типов rtwtypes (tensoraerospace/aerospacemodel/model/rtwtypes.py)

  .. code-block:: python

    class ExtY(ctypes.Structure):

    _fields_ = [
        ("name1", type_from_rtwtypes),
        ("name2", type_from_rtwtypes),
    ]

    Имя и тип можно посмотреть в сгенерированном С файле. Файл должен называться MODEL_NAME.h. В данном файле найдите описание External inputs, External outputs

  В dll файле существуют 3 функции
    * MODEL_NAME_initialize - служит для инициализации модели
    * MODEL_NAME_step - служит для расчета модели на следующем шаге модели
      шаг модели равен dt, определенном в параметрах Simulink модели
    * MODEL_NAME_terminate - служит для освобождении ресурсов модели

Пример использования Simulink модели  с Python:

  .. code-block:: 
        import os
        import ctypes

        import matplotlib.pyplot as plt

        from rtwtypes import *

        class ExtY(ctypes.Structure):
            """
                Output parameters Simulink model
                (name, type)
            """
            _fields_ = [
                ("u", real_T),
                ("w", real_T),
                ("q", real_T),
                ("theta", real_T),
                ("sim_time", real_T),
            ]

            
        class ExtU(ctypes.Structure):
            """
                INput parameters Simulink model
                (name, type)
            """
            _fields_ = [
                ("ref_signal", real_T),
            ]


        dll_path = os.path.abspath("model.dll")
        dll = ctypes.cdll.LoadLibrary(dll_path)

        X = ExtU.in_dll(dll, 'uav1_model_U')
        Y = ExtY.in_dll(dll, 'uav1_model_Y')

        model_initialize = dll.model_initialize
        model_step = dll.model_step
        model_terminate = dll.model_terminate


        model_initialize()

        u = []
        w = []
        q = []
        theta = []

        for step in range(int(2100)):
            X.ref_signal = -0.0
            model_step()
            u.append(Y.u)
            w.append(Y.w)
            q.append(Y.q)
            theta.append(Y.theta)

        model_terminate()



        plt.plot(wz)

        plt.ylabel('$u$, [м/с]')

        plt.show()