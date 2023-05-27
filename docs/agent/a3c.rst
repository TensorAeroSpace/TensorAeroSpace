A3C
================================================================
Алгоритм Asynchronous Advantage Actor Critic использует advantage функцию для обновления агентов.
Также в данном алгоритме используются асинхронные агенты для исследования среды и в качестве замены реплей буфера.

.. autoclass:: tensorairspace.agent.a3c.model.Agent
  :members:
  :inherited-members:

.. autoclass:: tensorairspace.agent.a3c.model.Worker
  :members:
  :inherited-members:

.. autoclass:: tensorairspace.agent.a3c.model.Actor
  :members:
  :inherited-members:

.. autoclass:: tensorairspace.agent.a3c.model.Critic
  :members:
  :inherited-members:


Источники
---------
- `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_

С какими окружениями gym можно использовать?
--------------------------------------------
- Unity среда