# Метрики TensorBoard

> :material-chart-line: Унифицированная схема TensorBoard для всех RL-агентов в TensorAeroSpace.

Все RL-агенты в TensorAeroSpace записывают скаляры и гистограммы TensorBoard
через единый `MetricWriter`, пространство имён тегов которого определено в
`tensoraerospace.agent.metrics.schema`. Поскольку каждый агент использует
одинаковые имена, общую ось (накопленные шаги среды) и одинаковый минимальный
набор метрик, запуски PPO, SAC, DQN, ADP, GAIL и других корректно
накладываются друг на друга на одном графике TensorBoard.

!!! note
    Старые запуски, записанные с предыдущим, специфичным для конкретного
    агента, наименованием не будут совмещаться с новыми запусками на одном
    графике. Унификация — это жёсткое переименование, легаси-карта алиасов
    была удалена.

## Зачем нужна единая схема

До унификации одна и та же величина логировалась под разными именами в
зависимости от агента — `Performance/Episode_Reward`, `Performance/Reward`,
`performance/episode_reward`, `episode_reward` и так далее. Префиксы для
loss-ов смешивались между `Loss/`, `loss/` и `losses/`. Ось X в одних
случаях была индексом эпизода, в других — накопленным числом шагов среды,
из-за чего сравнения агентов на одном графике становились некорректными.

Единая схема решает все три проблемы сразу:

- Одно каноническое имя на каждую величину (`lowercase_snake_case`, `/` как разделитель групп).
- Одна ось: `env_step` — обязательный аргумент в каждом вызове `add_scalar`.
- Один обязательный минимум, который должен логировать каждый RL-агент,
  чтобы любые два агента всегда были сравнимы хотя бы по базовым метрикам
  rollout/training.

## Префиксы групп

| Префикс | Назначение |
|---|---|
| `rollout/` | Поэпизодная статистика среды (награда, длина, общее число шагов). |
| `loss/` | Тренировочные потери (actor, critic, entropy, value, policy и специфичные для алгоритма). |
| `policy/` | Статистика политики / действий (entropy, std действий, среднее log-pi). |
| `value/` | Статистика value-функции (среднее V, TD-цели, TD-ошибки, advantages). |
| `diagnostics/` | Специфичные для алгоритма диагностики (KL, clip fraction, accuracy). |
| `train/` | Счётчики прогресса обучения (updates, learning rate, размер replay-буфера). |
| `eval/` | Статистика эпизодов оценки (награда, длина). |
| `weights/` | Гистограммы весов сетей (`weights/<group>/<param>`). |
| `grads/` | Гистограммы градиентов (`grads/<group>/<param>`). |

## Tier 1 — Обязательный минимум

Каждый RL-агент обязан логировать перечисленные ниже метрики. Их наличие
проверяется в конце `train()` функцией `assert_contract_satisfied()`.

| Константа | Тег | Примечания |
|---|---|---|
| `ROLLOUT_EPISODE_REWARD` | `rollout/episode_reward` | пишется в конце эпизода |
| `ROLLOUT_EPISODE_LENGTH` | `rollout/episode_length` | пишется в конце эпизода |
| `ROLLOUT_TOTAL_STEPS` | `rollout/total_steps` | накопленные шаги среды |
| `TRAIN_UPDATES` | `train/updates` | накопленные градиентные обновления |
| `TRAIN_LR` | `train/lr` | текущий learning rate (агенты с константным LR могут писать его один раз в начале) |

`EVAL_EPISODE_REWARD` (`eval/episode_reward`) и `EVAL_EPISODE_LENGTH`
(`eval/episode_length`) обязательны только если агент выполняет цикл оценки.

## Tier 2 — Общие константы

Эти метрики логируются, если соответствующая величина существует в
алгоритме. Они объявлены как константы уровня модуля в
`tensoraerospace.agent.metrics.schema`.

### `loss/*`

| Константа | Тег | Краткое описание |
|---|---|---|
| `LOSS_ACTOR` | `loss/actor` | Loss актора / policy gradient. |
| `LOSS_CRITIC` | `loss/critic` | Loss критика / value-функции для actor-critic методов. |
| `LOSS_ENTROPY` | `loss/entropy` | Член энтропийной регуляризации. |
| `LOSS_VALUE` | `loss/value` | Отдельный value loss (когда отличается от `loss/critic`). |
| `LOSS_POLICY` | `loss/policy` | Policy loss для off-policy actor-critic (SAC, DDPG). |

### `train/*`

| Константа | Тег | Краткое описание |
|---|---|---|
| `TRAIN_REPLAY_SIZE` | `train/replay_size` | Текущая заполненность replay-буфера (off-policy). |

### `policy/*`

| Константа | Тег | Краткое описание |
|---|---|---|
| `POLICY_ENTROPY` | `policy/entropy` | Дифференциальная энтропия текущей политики. |
| `POLICY_ACTION_STD` | `policy/action_std` | Среднее по размерностям стандартное отклонение распределения действий. |
| `POLICY_ACTION_ABS_MEAN` | `policy/action_abs_mean` | Среднее по абсолютной величине действий (индикатор насыщения). |

### `value/*`

| Константа | Тег | Краткое описание |
|---|---|---|
| `VALUE_MEAN` | `value/mean` | Среднее V(s) по батчу. |
| `VALUE_TD_TARGET` | `value/td_target_mean` | Среднее TD-цели. |
| `VALUE_TD_ERROR_MEAN` | `value/td_error_mean` | Средняя TD-ошибка (target − prediction). |
| `VALUE_TD_ERROR_MAX` | `value/td_error_max` | Максимальная TD-ошибка в батче. |
| `VALUE_TD_ERROR_MIN` | `value/td_error_min` | Минимальная TD-ошибка в батче. |

### `diagnostics/*`

| Константа | Тег | Краткое описание |
|---|---|---|
| `DIAG_TERMINATED_COUNT` | `diagnostics/terminated_count` | Число завершённых (terminated) эпизодов с момента последнего лога. |
| `DIAG_TRUNCATED_COUNT` | `diagnostics/truncated_count` | Число оборванных (truncated) эпизодов с момента последнего лога. |

### `eval/*`

| Константа | Тег | Краткое описание |
|---|---|---|
| `EVAL_EPISODE_REWARD` | `eval/episode_reward` | Награда эпизода оценки. |
| `EVAL_EPISODE_LENGTH` | `eval/episode_length` | Длина эпизода оценки. |

## Tier 3 — Специфичные для алгоритма метрики

У каждого алгоритма есть собственный класс с пространством имён внутри
`schema.py`. Теги по-прежнему используют общие префиксы групп, поэтому
группы TensorBoard остаются консистентными между запусками.

### PPO

`from tensoraerospace.agent.metrics.schema import PPO`

| Константа | Тег | Краткое описание |
|---|---|---|
| `PPO.APPROX_KL` | `diagnostics/approx_kl` | Эмпирический KL между старой и новой политикой. |
| `PPO.CLIP_FRACTION` | `diagnostics/clip_fraction` | Доля сэмплов, попадающих в границу clip. |
| `PPO.EXPLAINED_VARIANCE` | `diagnostics/explained_variance` | 1 − Var(returns − V) / Var(returns). |
| `PPO.REWARD_MEDIAN` | `rollout/episode_reward_median` | Медиана награды эпизодов в окне rollout. |
| `PPO.REWARD_P10` | `rollout/episode_reward_p10` | 10-й перцентиль наград эпизодов (нижняя граница). |
| `PPO.REWARD_P90` | `rollout/episode_reward_p90` | 90-й перцентиль наград эпизодов (верхняя граница). |

### SAC

`from tensoraerospace.agent.metrics.schema import SAC`

| Константа | Тег | Краткое описание |
|---|---|---|
| `SAC.LOSS_Q1` | `loss/q1` | Loss первой Q-сети. |
| `SAC.LOSS_Q2` | `loss/q2` | Loss второй Q-сети. |
| `SAC.LOSS_ALPHA` | `loss/alpha` | Loss параметра температуры. |
| `SAC.ALPHA_VALUE` | `policy/alpha` | Текущая температура энтропии α. |
| `SAC.Q_MEAN` | `value/q_mean` | Среднее Q(s, a) по батчу. |
| `SAC.LOG_PI_MEAN` | `policy/log_pi_mean` | Среднее log π(a|s) для сэмплированных действий. |

SAC также использует общие константы `LOSS_POLICY` (`loss/policy`) и
`TRAIN_REPLAY_SIZE` (`train/replay_size`).

### DSAC

`from tensoraerospace.agent.metrics.schema import DSAC`

DSAC наследует все имена SAC и добавляет члены регуляризации CAPS.

| Константа | Тег | Краткое описание |
|---|---|---|
| `DSAC.CAPS_SPATIAL` | `loss/caps_spatial` | Пространственный штраф гладкости CAPS. |
| `DSAC.CAPS_TEMPORAL` | `loss/caps_temporal` | Временной штраф гладкости CAPS. |

### DQN

`from tensoraerospace.agent.metrics.schema import DQN`

| Константа | Тег | Краткое описание |
|---|---|---|
| `DQN.LOSS_Q` | `loss/q` | Loss Беллмана / Хьюбера на Q-значениях. |
| `DQN.Q_PRED_SA_MEAN` | `value/q_pred_mean` | Среднее предсказанное Q(s, a). |
| `DQN.Q_TARGET_SA_MEAN` | `value/q_target_mean` | Среднее target Q(s, a). |
| `DQN.EPSILON` | `train/epsilon` | Текущее ε для ε-greedy исследования. |
| `DQN.PER_BETA` | `train/per_beta` | Показатель importance-sampling β для prioritized replay. |
| `DQN.TARGET_UPDATE` | `train/target_update` | Счётчик обновлений target-сети. |

DQN также использует общую константу `TRAIN_REPLAY_SIZE` (`train/replay_size`).

### DDPG

`from tensoraerospace.agent.metrics.schema import DDPG`

У DDPG нет собственных DDPG-специфичных тегов. Он использует только общие
константы Tier 2: `LOSS_POLICY` (`loss/policy`), `LOSS_VALUE` (`loss/value`)
и `TRAIN_REPLAY_SIZE` (`train/replay_size`).

### A2C

`from tensoraerospace.agent.metrics.schema import A2C`

| Константа | Тег | Краткое описание |
|---|---|---|
| `A2C.ADVANTAGE_MEAN` | `value/advantage_mean` | Среднее advantage до нормализации. |
| `A2C.ADVANTAGE_STD` | `value/advantage_std` | Std advantage до нормализации. |
| `A2C.ADVANTAGE_NORMALIZED_MEAN` | `value/advantage_normalized_mean` | Среднее нормализованного advantage. |
| `A2C.VALUE_BEFORE_UPDATE` | `value/before_update_mean` | Среднее V(s) до градиентного обновления. |
| `A2C.ENTROPY_BETA` | `policy/entropy_beta` | Текущий коэффициент энтропийной регуляризации β. |

### ADP

`from tensoraerospace.agent.metrics.schema import ADP`

| Константа | Тег | Краткое описание |
|---|---|---|
| `ADP.DHP_PHASE_EPISODE` | `train/dhp_phase_episode` | Индекс эпизода внутри текущей фазы DHP. |
| `ADP.LOSS_ACTOR_HDP` | `loss/actor_hdp` | Loss актора в HDP-варианте. |
| `ADP.LOSS_ACTOR_GDHP` | `loss/actor_adgdhp` | Loss актора в AD-GDHP варианте. |
| `ADP.LOSS_CRITIC_HDP` | `loss/critic_hdp` | Loss критика в HDP-варианте. |
| `ADP.LOSS_CRITIC_GDHP` | `loss/critic_gdhp` | Loss критика в GDHP-варианте. |
| `ADP.LOSS_CRITIC_LAMBDA` | `loss/critic_lambda` | Loss критика, взвешенный по λ. |

### ADHDP

`from tensoraerospace.agent.metrics.schema import ADHDP`

| Константа | Тег | Краткое описание |
|---|---|---|
| `ADHDP.DO_CRITIC` | `train/do_critic` | 1, если критик обновлялся на этом шаге, иначе 0. |
| `ADHDP.DO_ACTOR` | `train/do_actor` | 1, если актор обновлялся на этом шаге, иначе 0. |
| `ADHDP.ACTION_SAT_FRAC` | `policy/action_sat_frac` | Доля действий, попадающих в границы насыщения. |

### GAIL

`from tensoraerospace.agent.metrics.schema import GAIL`

| Константа | Тег | Краткое описание |
|---|---|---|
| `GAIL.LOSS_DISCRIMINATOR` | `loss/discriminator` | BCE-loss дискриминатора. |
| `GAIL.LOSS_GENERATOR` | `loss/generator` | Loss генератора (политики) против оценки дискриминатора. |
| `GAIL.EXPERT_ACCURACY` | `diagnostics/expert_accuracy` | Точность дискриминатора на экспертных переходах. |
| `GAIL.POLICY_ACCURACY` | `diagnostics/policy_accuracy` | Точность дискриминатора на переходах политики. |

## Конвенция для multi-worker

Для агентов с несколькими параллельными воркерами (в первую очередь A3C)
поворкерные скаляры используют **суффикс** `/worker_<id>`, чтобы префикс
группы оставался общим, и TensorBoard размещал всех воркеров в одной группе.

```text
rollout/episode_reward/worker_0
rollout/episode_reward/worker_1
loss/actor/worker_0
loss/actor/worker_1
```

`MetricWriter` валидирует такие теги, отрезая хвостовой сегмент
`/worker_<N>` перед проверкой по реестру. Любой зарегистрированный тег
скаляра можно дополнить суффиксом `/worker_<N>`, где `<N>` —
неотрицательное целое.

## Конвенция для гистограмм

Гистограммы используют выделенные группы верхнего уровня (`weights/`,
`grads/`) с двумя уровнями вложенности:

```text
weights/actor/<param_name>
weights/critic/<param_name>
grads/actor/<param_name>
grads/critic/<param_name>
```

`MetricWriter.add_histogram` валидирует первые два сегмента. Допустимые
группы верхнего уровня (`HISTOGRAM_GROUPS`) — `weights` и `grads`.
Допустимые подгруппы (`HISTOGRAM_SUBGROUPS`):

| Подгруппа | Используется в |
|---|---|
| `actor` | actor-critic агенты (PPO, A2C, DDPG, SAC, DSAC, ADP, ADHDP) |
| `critic` | actor-critic агенты (PPO, A2C, DDPG, SAC, DSAC, ADP, ADHDP) |
| `policy` | агенты только с политикой и политики SAC-стиля |
| `value` | отдельные value-сети |
| `q` | Q-сеть DQN |
| `q1`, `q2` | сдвоенные Q-сети SAC / DSAC |
| `discriminator` | дискриминатор GAIL |

## Пример end-to-end

```python
from tensoraerospace.agent.metrics import MetricWriter, schema
from tensoraerospace.agent.metrics.schema import (
    LOSS_ACTOR,
    LOSS_CRITIC,
    POLICY_ENTROPY,
    PPO,
)

# Создаём writer со строгим whitelist и тегом алгоритма.
self.writer = MetricWriter(log_dir=self.log_dir, algo="ppo")

# Внутри update() — каждый скаляр требует env_step.
self.writer.add_scalar(LOSS_ACTOR,    actor_loss,  env_step=self.global_step)
self.writer.add_scalar(LOSS_CRITIC,   critic_loss, env_step=self.global_step)
self.writer.add_scalar(POLICY_ENTROPY, entropy,    env_step=self.global_step)

# Специфичные для алгоритма метрики (PPO).
self.writer.add_scalar(PPO.APPROX_KL,     kl,          env_step=self.global_step)
self.writer.add_scalar(PPO.CLIP_FRACTION, clip_frac,   env_step=self.global_step)

# Гистограммы (валидируются правилом weights/<group>/<param>).
for name, p in self.actor.named_parameters():
    self.writer.add_histogram(f"weights/actor/{name}", p, env_step=self.global_step)
    if p.grad is not None:
        self.writer.add_histogram(f"grads/actor/{name}", p.grad, env_step=self.global_step)

# В конце эпизода — атомарная запись обязательного минимума rollout/*.
self.writer.log_episode(
    reward=ep_reward,
    length=ep_length,
    env_step=self.global_step,
    terminated=terminated,
    truncated=truncated,
)

# В конце train() — громко падаем, если какой-то обязательный тег так и не был записан.
self.writer.assert_contract_satisfied()
self.writer.close()
```

!!! tip
    `env_step` — обязательный именованный аргумент в каждом вызове
    `add_scalar` и `add_histogram`. Это вынуждает каждого агента
    задумываться об оси X и гарантирует корректное наложение запусков на
    общих графиках.

## Бэкенд Wandb

`MetricWriter` поддерживает Weights & Biases как второй sink наряду с
TensorBoard. Оба бэкенда могут быть активны одновременно — каждый вызов
`add_scalar` / `add_histogram` / `log_episode` рассылается во все
включённые sink-и.

### Когда wandb активен

| `tb_log_dir` | `wandb_project` | `WANDB_API_KEY` | TB | wandb |
|---|---|---|---|---|
| set | — | — | ✅ | — |
| set | — | set | ✅ | ✅ (project = `algo`) |
| — | — | set | — | ✅ (project = `algo`) |
| set | set | * | ✅ | ✅ (project как задан) |
| — | set | unset | — | ✅ (вызывает `wandb.login()` интерактивно) |
| — | — | unset | — | — |

### Пример

```python
from tensoraerospace.agent.sac.sac import SAC

# Только TensorBoard (по умолчанию)
agent = SAC(env=env, log_dir="runs/sac")

# Только wandb — установите WANDB_API_KEY в окружении, затем:
agent = SAC(
    env=env,
    wandb_project="my-experiment",
    wandb_tags=["sac", "pendulum"],
)

# Оба бэкенда параллельно
agent = SAC(
    env=env,
    log_dir="runs/sac",
    wandb_project="my-experiment",
    wandb_config={"lr": 3e-4, "tau": 5e-3},
)
```

### Параметры агента (kwargs)

`__init__` каждого RL-агента принимает следующие именованные аргументы
наряду с `log_dir`:

| Kwarg | Тип | Описание |
|---|---|---|
| `wandb_project` | `str \| None` | Имя проекта wandb. По умолчанию равно `algo`, если установлен `WANDB_API_KEY`. |
| `wandb_entity` | `str \| None` | Пространство имён команды/пользователя wandb. По умолчанию используется то, что выбирает сам `wandb`. |
| `wandb_run_name` | `str \| None` | Отображаемое имя запуска. По умолчанию `<algo>-<timestamp>`. |
| `wandb_tags` | `list[str] \| None` | Теги, привязанные к запуску. По умолчанию `[algo]`. |
| `wandb_config` | `dict \| None` | Словарь гиперпараметров, сохраняемый wandb вместе с запуском. |

### Аутентификация

Есть три способа аутентификации в wandb:

1. **CLI `wandb login`** (одноразовая настройка) — запускается интерактивно, сохраняет API-ключ в `~/.netrc`, чтобы последующие запуски подхватывали его автоматически.
2. **Переменная окружения `WANDB_API_KEY`** — самый простой вариант для CI; ключ читается на этапе инициализации sink-а. Если одновременно присутствуют `~/.netrc` и переменная окружения, побеждает переменная окружения.
3. **Интерактивный запрос** — если ключ не найден, а `wandb_project` передан явно, `_WandbSink` вызывает `wandb.login()`, который открывает интерактивный запрос (или вкладку браузера) для входа.

В CI без TTY интерактивный запрос завершится ошибкой — задайте `WANDB_API_KEY` как секрет пайплайна.

### Офлайн-режим

Чтобы запускать wandb без отправки данных (медленные соединения, изолированные сети, отладка без засорения облачного проекта):

```bash
export WANDB_MODE=offline
```

Sink всё равно создаст локальную директорию запуска внутри `wandb/`. Синхронизировать её позже можно командой:

```bash
wandb sync wandb/offline-run-*
```

Чтобы полностью отключить wandb так, чтобы даже локальная директория запуска не создавалась, не задавайте `WANDB_API_KEY` и не передавайте `wandb_project`.

### Отслеживание гиперпараметров

Используйте `wandb_config`, чтобы сохранять гиперпараметры обучения вместе с запуском. UI wandb позволяет затем сортировать, фильтровать и группировать запуски по значениям гиперпараметров:

```python
agent = SAC(
    env=env,
    wandb_project="sac-tuning",
    wandb_config={
        "lr": 3e-4,
        "tau": 5e-3,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "automatic_entropy_tuning": True,
    },
)
```

Всё, что передано в `wandb_config`, сохраняется как плоский снимок «ключ-значение» вместе с запуском (отображается в панели «Config» UI wandb) — отдельно от метрик-временных рядов. Снимок делается в момент `wandb.init()`; последующие изменения словаря не отражаются.

### Группировка связанных запусков

Для экспериментов с несколькими сидами или sweep-ов по гиперпараметрам используйте переменную окружения `WANDB_RUN_GROUP`, чтобы сгруппировать связанные запуски в UI wandb:

```bash
export WANDB_RUN_GROUP="sac-pendulum-seed-sweep"

for SEED in 0 1 2 3 4; do
    python train_sac.py --seed=$SEED
done
```

Все пять запусков появятся в одной группе в UI wandb; представление группы агрегирует метрики медианой и полосами перцентилей.

### Рекомендации

- **Всегда задавайте `wandb_config`** для любого запуска, который вы потенциально захотите сравнивать позже. Два запуска с одинаковыми метриками, но без конфигурации, неразличимы в UI.
- **Используйте `wandb_tags` как таксономию** — `["sac", "pendulum", "ablation-no-target-update"]` вместо того, чтобы упихивать всё в имя запуска. Теги фильтруются в UI.
- **Не помещайте секреты в `wandb_config`.** Он загружается как есть и виден всем, у кого есть доступ к проекту.
- **Фиксируйте `wandb_run_name`** для запусков, на которые вы будете ссылаться в статьях или отчётах. Автогенерируемые имена wandb (`prosperous-sea-42`) запоминаются, но непригодны для поиска; детерминированные имена вида `sac-pendulum-seed-3-2026-04-20` проще цитировать.

### Диагностика

- **`wandb.errors.UsageError: api_key not configured`** — ваш `wandb login` отработал на другой машине/пользователе. Запустите `wandb login --relogin` или задайте `WANDB_API_KEY`.
- **Запуск зависает на «Waiting for wandb.init()»** — обычно это сетевая проблема или фаервол. Попробуйте `export WANDB_MODE=offline`, чтобы убедиться, что само обучение работает, и синхронизировать данные позже.
- **Несколько агентов в одном Python-процессе пишут в чужие запуски** — этого происходить НЕ должно. Каждый `_WandbSink` пишет через свой захваченный объект run (`self._run.log(...)`), а не через модульное глобальное состояние `wandb.log(...)`. Если вы наблюдаете такое поведение — заведите issue.
- **Воркеры A3C не появляются в wandb** — это by design (см. раздел про ограничение A3C ниже). Воркеры (форкнутые) пропускают инициализацию wandb даже при заданном `WANDB_API_KEY`. Чтобы получить отдельные wandb-запуски на каждого воркера, запускайте по одному процессу на воркер вне фреймворка и используйте `WANDB_RUN_GROUP`, чтобы держать их вместе в UI.

### Ограничение A3C

A3C запускает воркеров в форкнутых процессах. Sink wandb инициализируется
только в главном процессе — воркеры по-прежнему делят с родителем общий
event-файл TensorBoard через суффикс `/worker_<id>`. Чтобы получить
отдельные wandb-запуски на каждого воркера, запускайте по одному процессу
на воркер вне фреймворка (каждый со своим `WANDB_RUN_GROUP`), а не через
прямой вызов `Agent.train()`.

## Как добавить новый алгоритм

1. **Отредактируйте `tensoraerospace/agent/metrics/schema.py`.** Добавьте
   новый класс с именем по алгоритму:

   ```python
   class MyAlgo:
       MY_LOSS = "loss/my_term"
       MY_DIAG = "diagnostics/my_signal"
   ```

   Переиспользуйте существующие префиксы групп (`loss/`, `policy/`,
   `value/`, `train/`, `diagnostics/`, `rollout/`, `eval/`), чтобы группы
   TensorBoard оставались консистентными. Теги должны быть в
   `lowercase_snake_case` с `/` в качестве разделителя групп и без
   пробелов.

2. **Зарегистрируйте класс в `_build_registry()`.** Добавьте `MyAlgo` в
   кортеж классов, перебираемых внутри `_build_registry()`, чтобы его
   константы попали в `REGISTRY`:

   ```python
   for cls in (PPO, SAC, DSAC, DQN, DDPG, A2C, ADP, ADHDP, GAIL, MyAlgo):
       parts.append(_collect_constants(cls))
   ```

3. **Используйте константы из вашего агента.** Импортируйте их по имени;
   никогда не используйте свободные строковые литералы для тегов:

   ```python
   from tensoraerospace.agent.metrics.schema import MyAlgo
   self.writer.add_scalar(MyAlgo.MY_LOSS, loss, env_step=self.global_step)
   ```

4. **Запустите unit-тесты схемы и writer-а.** Они выявят дублирующиеся
   значения тегов, некорректно сформированные имена и любые новые
   константы, отсутствующие в `REGISTRY`.

## Справочные ссылки

- Канонический источник: `tensoraerospace/agent/metrics/schema.py`
- Реализация writer-а: `tensoraerospace/agent/metrics/writer.py`
- Хелпер контракта: `tensoraerospace/agent/metrics/contract.py`
