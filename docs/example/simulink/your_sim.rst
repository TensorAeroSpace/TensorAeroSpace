Интеграция собственной Simulink модели
======================================

Генерация C++ кода
------------------

Для поддержки ОУ из ПО Simulink необходима надстройка для Simulink – Embedded Coder.

Для преобразования Simulink модели в С код:

#. При помощи блоков In1/Out1 опишите входные и выходные параметры

#. 	В настройках Simulink выберите: Code Generation/System target file ert_shrlib.tlc.
	
	.. image:: img/cpp_gen.png
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

  где *.c - все файлы с расширением c

  В папке появится so файл.

#. Опишите интерфейс взаимодействия

  Интерфейс взаимодействия описывается для входных и выходных параметров при помощи ctypes.Structure и преобразователя типов rtwtypes (tensorairspace/aerospacemodel/model/rtwtypes.py)

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

Пример использования Simulink модели Ту с Python:

Модель находится в https://github.com/TensorAirSpace/simulink-example

	.. image:: img/model.png
  		:width: 400
  		:alt: Модель

.. container:: cell code

   .. code:: python

      import os
      import ctypes

      import matplotlib.pyplot as plt

      from rtwtypes import *

.. container:: cell code

   .. code:: python

      class ExtY(ctypes.Structure):
          """
              Output parameters Simulink model
              (name, type)
          """
          _fields_ = [
              ("Wz", real_T),
              ("theta_big", real_T),
              ("H", real_T),
              ("alpha", real_T),
              ("theta_small", real_T),
          ]

          
      class ExtU(ctypes.Structure):
          """
              INput parameters Simulink model
              (name, type)
          """
          _fields_ = [
              ("ref_signal", real_T),
          ]

.. container:: cell code

   .. code:: python

      dll_path = os.path.abspath("model.so")
      dll = ctypes.cdll.LoadLibrary(dll_path)

.. container:: cell code

   .. code:: python

      X = ExtU.in_dll(dll, 'model_U')
      Y = ExtY.in_dll(dll, 'model_Y')

.. container:: cell code

   .. code:: python

      model_initialize = dll.model_initialize
      model_step = dll.model_step
      model_terminate = dll.model_terminate

.. container:: cell code

   .. code:: python

      model_initialize()

      wz = []
      theta_big = []
      H = []
      alpha = []
      theta_small = []

      for step in range(int(2100)):
          X.ref_signal = -0.1
          model_step()
          
          wz.append(Y.Wz)
          theta_big.append(Y.theta_big)
          H.append(Y.H)
          alpha.append(Y.alpha)
          theta_small.append(Y.theta_small)

      model_terminate()

   .. container:: output execute_result

      ::

         0

.. container:: cell code

   .. code:: python

      plt.plot(wz)

      plt.ylabel('$w_z$, [рад/с]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$w_z$, [рад/с]')

   .. container:: output display_data

      .. image:: img/wz.png

.. container:: cell code

   .. code:: python

      plt.plot(H)

      plt.ylabel('H, [м]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, 'H, [м]')

   .. container:: output display_data

      .. image:: img/h.png

.. container:: cell code

   .. code:: python

      plt.plot(theta_big)

      plt.ylabel('$\Theta$, [рад]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$\\Theta$, [рад]')

   .. container:: output display_data

      .. image:: img/theta_big.png

.. container:: cell code

   .. code:: python

      plt.plot(theta_small)

      plt.ylabel(r'$\theta$, [рад]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$\\theta$, [рад]')

   .. container:: output display_data

      .. image:: img/theta_small.png

.. container:: cell code

   .. code:: python

      plt.plot(alpha)

      plt.ylabel(r'$\alpha$, [рад]')

   .. container:: output execute_result

      ::

         Text(0, 0.5, '$\\alpha$, [рад]')

   .. container:: output display_data

      .. image:: img/alpha.png
