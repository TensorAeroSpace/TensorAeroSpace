Generative Adversarial Imitation Learning (GAIL)
================================================================

Generative Adversarial Imitation Learning (GAIL) - Алгоритм иммитационного обучения который использует сеть дискриминатор для оценки качества действий агента.

Как работает GAIL
----------------
GAIL обучается выдавать такие действия чтобы дискриминатор не мог их отличить от экспертных


Документация
------------

.. autoclass:: tensoraerospace.agent.gail.model.GAIL
  :members:


Источники
---------
- `Generative Adversarial Imitation Learning <https://arxiv.org/pdf/1606.03476>`_

На каких средах протестили:
--------------------------------------------
- Unity среда