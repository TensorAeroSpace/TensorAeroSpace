Создание модели в Simulink
=========================

Создание объекта управления в симулинке
---------------------------------------

.. contents:: На этой странице
   :local:
   :depth: 1

.. image:: img/image017.png
  :width: 400
  :alt: ОУ в ПО Simulink


Для создания ОУ в ПО Simulink:

#. В рабочее поле добавьте элементы из библиотеки Simulink:

        * Simulink/Continuous/State-Space

        * Simulink/Sources/Digital Clock

        * Simulink/Comonly Used Block/In1

        * Simulink/Comonly Used Block/Out1

#. Переименуйте блоки In1/Out1 в осмысленные имена сигналов.

#. В блоке State-Space задайте параметры (удобно через MATLAB scripts)

	.. image:: img/image018.png
  		:width: 400
  :alt: Блок State-Space