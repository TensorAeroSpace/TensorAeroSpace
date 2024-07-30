.. tensoraerospace documentation master file, created by
   sphinx-quickstart on Wed Feb  9 19:09:22 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Добро пожаловать в документацию TensorAeroSpace! - RL for Aerospace
===================================================================

TensorAeroSpace - это набор объектов управления, сред моделирования OpenAI Gym и реализации алгоритмов Reinforcement Learning (RL)

Github репозиторий: https://github.com/TensorAeroSpace/TensorAeroSpace


.. toctree::
   :maxdepth: 2
   :caption: User guide:

   guide/installation.rst
   guide/unity_env.rst
  

.. toctree::
   :maxdepth: 2
   :caption: Объекты управления:

   model/f16
   model/b747
   model/EVL
   model/typical_rocket
   model/comsat
   model/f4c
   model/geosat
   model/uav
   model/x15
   model/lsu
   model/suav
   model/unity_env.rst

.. toctree::
   :maxdepth: 3
   :caption: Агенты:

   agent/ihdp
   agent/mpc
   agent/dqn
   agent/a3c
   agent/a2c_narx
   agent/ppo
   agent/sac
   agent/ddpg
   agent/gail
   optimization/optuna_based.rst
   benchmark/metrics.rst
   benchmark/bench.rst
   signals/signals.rst


.. toctree::
   :maxdepth: 2
   :caption: Примеры:
   
   example/env/examples.rst
   example/agent/ihdp/example_ihdp.rst
   example/agent/sac/example-sac-f16.rst
   example/optimization/example_optimization.rst
   example/agent/a2c_narx/example-narx.rst
   example/agent/mpc/example_mpc.rst
   example/agent/mpc/example_mpc_stochastic.rst
   example/failure/ihdp-failure.rst
   example/simulink/sim_pyth.rst
   example/simulink/your_sim.rst
   example/env/unity_example.rst
   example/simintech/simintech.rst
   example/simintech/simintechToPython.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
