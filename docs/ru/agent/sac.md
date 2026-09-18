# Soft Actor‑Critic (SAC)

!!! note "Переходы векторной среды"
    `train_vector()` выбирает начальные случайные действия в границах среды. При автосбросе `info["final_observation"]` и необязательная маска `info["_final_observation"]` сохраняют последнее наблюдение для replay buffer; оценка будущей награды обнуляется только при настоящем завершении. Для старых сред без этих данных сохранена консервативная маска завершения. `ImprovedB747VecEnvTorch` передаёт оба поля.

SAC — off‑policy актор‑критик с максимизацией энтропии: обучает стохастическую политику, одновременно повышая ожидаемую награду и энтропию (исследовательность). В нашей реализации используются сдвоенные Q‑сети, target‑критик, гауссовская/детерминированная политика, реплей‑буфер, soft‑обновление и опциональная автонастройка энтропии.

![SAC схема](../agent/img/sac/sac.png){ width=800 }

## Сбор переходов в скалярной среде

`train()` копирует наблюдение до `env.step()`: повторное использование средой
одного изменяемого массива больше не портит переход в replay buffer.
При автоматическом сбросе среды действительное конечное состояние берётся
из `final_observation` или `terminal_observation`. Только истинное завершение
обнуляет продолжение в целевой оценке; пользовательский лимит шагов отмечается
как усечение. Короткие запуски до накопления обучающего батча явно записывают
ноль обновлений оптимизатора и проходят проверку обязательных метрик.

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

## Продолжение обучения и сохранение настроек

`train()` и `train_vector()` сохраняют `total_updates` и `total_env_steps` между
вызовами. Интервал обновления целевой сети отсчитывается по общему числу
градиентных обновлений. Разбиение scalar-обучения на вызовы по целым эпизодам
сохраняет результат при одинаковом потоке случайных чисел и данных среды.
`train()` возвращает число обновлений и лучшую награду **за текущий вызов**;
`best_reward` вычисляется и при `save_best=False`.

`policy_lr` задаёт скорость обучения Gaussian- и Deterministic-политик независимо
от `lr` критика. Checkpoint сохраняет обе скорости, частоту логирования и общие
счётчики. Старые checkpoint без счётчиков загружаются с нулями. Replay buffer,
состояние среды и генераторов случайных чисел не сохраняются: загрузка сохраняет
фазу обновления target-сети, но не воспроизводит обучение побитно.

Каждый вызов `train_vector()` сбрасывает среду и применяет собственный бюджет
`warmup_steps`. Для сравнения разбиений vector-обучения нужно согласовать границы
эпизодов и прогрев. При ручных вызовах `update_parameters(..., updates)` индексом
обновления управляет вызывающий код.

### Оценка без изменения обучения

`select_action(..., evaluate=True)` и `select_action_batch(..., evaluate=True)`
вычисляют среднее действие без генерации шума. Они не расходуют состояние RNG
PyTorch, не меняют буфер шума Deterministic-политики и не строят граф градиентов.
Поэтому вызов оценки сам по себе не меняет последующую тренировку. Если отдельная
оценочная среда использует глобальный RNG, её случайность нужно изолировать
самостоятельно. При `evaluate=False` исследовательское поведение сохраняется.

### Шум детерминированной политики

`policy_type="Deterministic"` генерирует независимый шум для каждой строки batch
и каждой компоненты действия: σ=0,1 в нормализованных координатах, ограничение
шума ±0,25. Затем шум умножается на половину диапазона соответствующего действия.
Результат ограничивается точными границами пространства действий. Шум поэтому
масштабируется одинаково для радиан, градусов и ньютонов. Старый буфер `noise`
сохранён для загрузки checkpoint, но больше не используется при генерации.

Оценка остаётся детерминированной. Исправление меняет последовательность
исследовательских действий и обучение; оно не гарантирует улучшения награды.

### Плотность гауссовой политики у пределов руля

Гауссова политика SAC вычисляет поправку Якобиана `tanh` по значению до
ограничения через устойчивую формулу с softplus и отдельно учитывает масштаб
действия. Градиент энтропии сохраняется, когда float32 округляет `tanh` до ±1.
Интервалы действий должны иметь конечную положительную ширину. Параметры сети
и детерминированные действия прежних весов сохраняются, но логарифмы плотности
и дальнейшее обучение меняются. Такая численная поправка не гарантирует
устойчивого управления: нужны сравнения ошибки слежения и нарушений границ
для нескольких seed и горизонтов оценки.

Формула соответствует [TanhTransform в PyTorch](https://github.com/pytorch/pytorch/blob/main/torch/distributions/transforms.py)
и поправке плотности в [SAC](https://arxiv.org/abs/1801.01290).
