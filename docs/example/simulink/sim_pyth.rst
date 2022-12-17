Примеры работы с Simulink моделями
==================================

Генерация C++ кода
------------------

Для поддержки ОУ из ПО Simulink необходима надстройка для Simulink – Embedded Coder.

Для преобразования Simulink модели в С код:

#. 	В настройках Simulink были выбраны: Code Generation/System target file ert_shrlib.tlc.
	
	.. image:: img/cpp_gen.png
  		:width: 400
  		:alt: Блок Stae-Space

#. Для построения модели использовалось сочетание клавиш ctrl+B. Также это можно сделать в панели навигации, выбрав пункт “Build model”. В результате появлялась папка с кодом на языке C++ в директории, в которой находилась модель. 

Интегрирование Simulink модели в Python 
---------------------------------------

Интегрирование Simulink модели в Python осуществлялось с помощью DLL библиотеки (библиотеки динамической компоновки). В результате генерации cpp кода получили код Simulink модели и Makefile с расширением .mk. После его запуска была скомпилирована DLL библиотека.

Для согласования типов данных и корректной работы модели использовался модуль ctypes и был написан преобразователь типов. tensorairspace/aerospacemodel/model/rtwtypes.py

Пример использования Simulink модели с Python хранится в example/example_sim_pyth.ipynb



Создание объекта управления в симулнке
---------------------------------------

.. image:: img/sim.png
  :width: 400
  :alt: ОУ в ПО Simulink


Для создания ОУ в ПО Simulink:

#. В рабочее поле были добавлены элементы из библиотеки Simulink:

        * Simulink/Continuous/State-Space

        * Simulink/Sources/Digital Clock

        * Simulink/Comonly Used Block/In1

        * Simulink/Comonly Used Block/Out1

#. Блоки In1/Out1 были переименованы в соответствующие названия.

#. В State-Space были заданы следующие параметры (для удобства работы использовали MATLAB Scripts)

	.. image:: img/sim_ss.png
  		:width: 400
  		:alt: Блок Stae-Space

#. Был создан MATLAB Script со следующим кодом:

    .. code-block:: matlab

        flag = 1;

        % Инициализцаия параметров
        [A,B,C,D] = b747_model(flag);

        init = [0 -0.0 -0.0 0];
        ref_signal = -0.10;

        % Время начала/конца/шага времени моделирования
        t_s = 0;
        t_e = 500;
        dt = 0.1;

        % Запуск Simulink модели
        simOut = sim('aircraft_sim.slx');

        y = simOut.get('yout');

        u = y.getElement(1).Values.Data;
        w = y.getElement(2).Values.Data;
        q = y.getElement(3).Values.Data;
        theta = y.getElement(4).Values.Data;
        t = y.getElement(5).Values.Data;