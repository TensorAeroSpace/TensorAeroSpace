.. TensorAirSpace documentation master file, created by
   sphinx-quickstart on Wed Feb  9 19:09:22 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorAirSpace: a library for applied reinforcement learning in aerospace 
==========================================================================

Библиотека TensorAirSpace предоставляет возможность применения методов reinforcement learning к различным объектам управления.

.. toctree::
   :maxdepth: 2
   :caption: User guide:

   guide/installation.rst

.. toctree::
   :maxdepth: 2
   :caption: Объекты управления:

   model/f16
   model/b747
   model/EVL
   model/typical_rocket

.. toctree::
   :maxdepth: 2
   :caption: Агенты:

   agent/ihdp
   agent/dqn
   optimization/optuna_based.rst

.. toctree::
   :maxdepth: 2
   :caption: Сигналы:

   signals/signals.rst


.. toctree::
   :maxdepth: 2
   :caption: Примеры:
   
   example/env/LinearLongitudinalF16.rst
   example/agent/ihdp/example_ihdp.rst 
   example/optimization/example_optimization.rst
   example/failure/ihdp-failure.rst
   example/simulink/sim_pyth.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
