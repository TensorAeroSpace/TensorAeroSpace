# Soft Actor‑Critic (SAC)

SAC — off‑policy актор‑критик с максимизацией энтропии: обучает стохастическую политику, одновременно повышая ожидаемую награду и энтропию (исследовательность). В нашей реализации используются сдвоенные Q‑сети, target‑критик, гауссовская/детерминированная политика, реплей‑буфер, soft‑обновление и опциональная автонастройка энтропии.

![SAC схема](../agent/img/sac/sac.png){ width=800 }

## Компоненты

- Две Q‑сети: `QNetwork(state, action) -> (Q1, Q2)` и целевая `critic_target`
- Политика: `GaussianPolicy` (по умолчанию) или `DeterministicPolicy` (без энтропии)
- Реплей‑буфер: `ReplayMemory` для выборки батчей
- Soft‑обновление целевой сети: `soft_update(target, source, tau)`
- Автонастройка энтропии: оптимизация `alpha` к целевой энтропии \(H_{\text{target}} = -\dim(\mathcal{A})\)

## Теория (на базе реализации)

- «Мягкая» целевая оценка для Q (double Q + энтропия):

$$
\begin{aligned}
& a' \sim \pi_\theta(\cdot|s')\ ,\ \log \pi_\theta(a'|s'), \\
& Q_{\text{targ}}(s,a) = r + \gamma\, \big( \min(Q_1(s',a'), Q_2(s',a')) - \alpha\, \log \pi_\theta(a'|s') \big)
\end{aligned}
$$

- Обучение критиков (MSE к таргету): \(\mathcal{L}_{Q_i} = \mathbb{E}[(Q_i(s,a) - Q_{\text{targ}})^2]\)

- Обучение политики (репараметризация):

$$
\mathcal{L}_\pi = \mathbb{E}_{s\sim \mathcal{D},\ \epsilon\sim\mathcal{N}}\big[ \alpha\, \log \pi_\theta(f_\theta(\epsilon; s) | s) - Q_{\min}(s, f_\theta(\epsilon; s)) \big]
$$

- Автонастройка \(\alpha\) (опц.):

$$
\mathcal{L}_\alpha = -\,\mathbb{E}_{a\sim\pi}\big[\log \alpha\, (\log \pi_\theta(a|s) + H_{\text{target}})\big]\ ,\quad \alpha \leftarrow e^{\log \alpha}
$$

## Быстрый старт

```python
import gymnasium as gym
from tensoraerospace.agent.sac.sac import SAC

env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
agent = SAC(env,
            updates_per_step=1,
            batch_size=64,
            memory_capacity=100000,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            policy_type='Gaussian',
            target_update_interval=1,
            automatic_entropy_tuning=True,
            hidden_size=256,
            device='cpu')

agent.train(num_episodes=100)
agent.save('./runs')
```

!!! tip
    Для непрерывного пространства действий используйте `GaussianPolicy` с `automatic_entropy_tuning=True` — это стабилизирует степень исследовательности.

## Унифицированный интерфейс обучения

Все RL‑агенты TensorAeroSpace используют общую сигнатуру `train()`,
определённую в `BaseRLModel`:

```python
def train(
    self,
    num_episodes: int = 100,
    *,
    max_steps: Optional[int] = None,
    save_best: bool = False,
    save_path: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> dict
```

Для SAC через `**kwargs` принимаются следующие специфичные опции:

- `save_best_with_gradients` (`bool`): включать состояния оптимизаторов
  в чекпоинты лучших моделей.

Пример:

```python
stats = agent.train(
    num_episodes=100,
    max_steps=500,
    save_best=True,
    save_path='./runs/sac_best',
)
print(stats['best_reward'], len(stats['episode_rewards']))
```

## Практические советы

- Увеличивайте `batch_size` и `memory_capacity` для более стабильных градиентов
- `tau` в пределах 0.005–0.02 для мягкого обновления target‑сети
- Если политика детерминированная — установите `alpha=0` и отключите автонастройку
- При использовании `DeterministicPolicy` с `action_space=None` учтите, что `action_scale` и `action_bias` теперь являются `torch.Tensor` (а не Python‑числами)

!!! warning "Gymnasium 5-tuple API"
    Реализация использует современный 5‑элементный API `step` из Gymnasium:
    ```python
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ```
    Если вы переходите со старого кода с 4‑элементным API (`next_state, reward, done, info = env.step(action)`), убедитесь, что среда совместима с Gymnasium и возвращает 5‑элементный кортеж.

## Документация API

::: tensoraerospace.agent.sac.sac.SAC

::: tensoraerospace.agent.sac.replay_memory.ReplayMemory

::: tensoraerospace.agent.sac.model.QNetwork

::: tensoraerospace.agent.sac.model.GaussianPolicy

::: tensoraerospace.agent.sac.model.DeterministicPolicy
