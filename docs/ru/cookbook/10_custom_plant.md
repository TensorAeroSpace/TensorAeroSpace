# Рецепт 10 — Добавить свой объект управления

**Цель.** Расширить TensorAeroSpace новым самолётом / ракетой / БПЛА: написать динамику, обернуть её в Gymnasium-среду по соглашениям библиотеки, зарегистрировать, линеаризовать для warm-start адаптивных агентов.

**Связано.** [Рецепт 02](02_env_anatomy.md) — контракт, которому должна удовлетворять ваша новая среда.

## Короткая версия

Чтобы добавить объект, пишите **два класса и один вызов регистрации**:

1. **Функция динамики** `f(x, u, t, params) -> xdot` — чистый NumPy, без Gymnasium-зависимостей.
2. **Класс Gymnasium-среды**, обёртка над динамикой с интегратором, совместимый с контрактом `state_space`/`tracking_states`/`reference_signal`. На практике наследуйте `tensoraerospace.envs.base.BaseEnv` (или скопируйте одну из существующих сред, например `envs/b747/linear.py`).
3. `gym.register('<NewPlant>-v0', entry_point='...')` — чтобы `gym.make` его нашёл.

## 1. Функция динамики

Начните с непрерывной ОДУ в форме пространства состояний. Пример — игрушечная продольная модель:

```python
import numpy as np

def toy_plant_ode(x: np.ndarray, u: np.ndarray, t: float,
                  params: dict) -> np.ndarray:
    """xdot = f(x, u, t, params). Чистый NumPy, без побочных эффектов.

    x = [alpha, q], u = [stab]
    alpha_dot = -Z_alpha * alpha + q
    q_dot = M_alpha * alpha + M_q * q + M_delta * stab
    """
    alpha, q = float(x[0]), float(x[1])
    stab = float(u[0])
    Z_alpha = params['Z_alpha']
    M_alpha = params['M_alpha']
    M_q = params['M_q']
    M_delta = params['M_delta']
    return np.array([
        -Z_alpha * alpha + q,
        M_alpha * alpha + M_q * q + M_delta * stab,
    ], dtype=np.float64)
```

Держите `params` как обычный dict — он хорошо сериализуется в JSON для воспроизводимости и чекпойнтов.

## 2. Gymnasium-среда

Возьмите существующую среду как шаблон. Линейные F-16 и B747 — самые простые; нелинейная F-16 — правильный шаблон, если динамика нелинейная.

Класс среды обязан минимум выставлять:

```python
class ToyPlantEnv(gym.Env):
    def __init__(
        self,
        number_time_steps: int,
        dt: float = 0.01,
        initial_state: list[list[float]] = [[0.0], [0.0]],
        reference_signal: np.ndarray | None = None,
        state_space: list[str] = ['alpha', 'q'],
        tracking_states: list[str] = ['alpha'],
        output_space: list[str] | None = None,
        use_reward: bool = True,
        reward_func: Callable | None = None,
        integrator: str = 'euler',
        params: dict | None = None,
    ): ...

    def reset(self, *, seed=None, options=None):
        """Вернуть (obs, info)."""

    def step(self, action):
        """Вернуть (obs, reward, terminated, truncated, info)."""

    def get_state(self):
        """Опциональный helper из примеров — текущее внутреннее состояние."""
```

Храните полное внутреннее состояние как `self._x` (форма `(n_full_state,)`) и срезайте его для наблюдения по `state_space`. Храните `self.ref_signal = reference_signal`, чтобы пользователи и агенты могли его читать.

### Шаг интегрирования

Напишите небольшой `_rk4_step` или `_euler_step`, продвигающий `self._x` на один `dt` через вашу `toy_plant_ode`. Это единственное место, где имеет значение интегратор:

```python
def _euler_step(self, u):
    xdot = self.ode(self._x, u, self._t, self.params)
    self._x = self._x + self.dt * xdot
    self._t += self.dt
```

## 3. Регистрация

В конце файла среды или в `tensoraerospace/envs/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id='ToyPlant-v0',
    entry_point='my_package.envs:ToyPlantEnv',
    max_episode_steps=10_000,
)
```

После этого:

```python
import gymnasium as gym
import tensoraerospace       # вызов register() внутри пакета станет доступен
env = gym.make('ToyPlant-v0', number_time_steps=2000, ...)
```

## 4. Warm-start адаптивных агентов

Как только среда работает с PID, вы можете сразу bootstrap'ить любого адаптивного агента (IHDP / iADP / AA-INDI) **при условии линеаризации вокруг трима**.

### Найти трим

```python
from scipy.optimize import fsolve

def trim_residual(z):
    alpha, stab = z
    return toy_plant_ode(np.array([alpha, 0.0]), np.array([stab]), 0.0, params)

alpha_trim, stab_trim = fsolve(trim_residual, x0=[0.0, 0.0])
```

### Получить `G_init` через численный Jacobian или PE-возбуждение

Численно (одна строка):

```python
eps = 1e-4
x0 = np.array([alpha_trim, 0.0])
u0 = np.array([stab_trim])
G_cont = (toy_plant_ode(x0, u0 + eps, 0, params) - toy_plant_ode(x0, u0, 0, params)) / eps
G_discrete = G_cont * dt         # что нужно iADP / AA-INDI
```

Либо запустите 3-секундный мультисинус и подберите G из `Δx / Δu` (ноутбук iADP F-16 именно так и делает).

### Подключить к агенту

См. [Рецепт 06](06_online_adaptive.md) — общий паттерн warm-start.

## 5. Тесты

Каждая новая среда должна иметь минимум:

- **Round-trip тест** — `env.reset()` → цикл со случайным `env.step()` → траектория конечна, в пределах `observation_space`.
- **Тест трима** — из `(alpha_trim, 0)` при действии `stab_trim` состояние остаётся на месте (`|x[t+1] - x[t]| < 1e-6`).
- **Тест формы reference_signal** — передача ссылочного сигнала неправильной формы даёт понятную ошибку.

Копируйте паттерны из `tests/envs/` для существующих сред.

## 6. Публикация

Если ваша среда в отдельном pip-пакете — убедитесь, что `register()` выполняется при `import`. Самая чистая идиома:

```python
# my_package/__init__.py
from . import envs  # триггерит регистрацию
```

Пользователи потом делают `import my_package; gym.make('MyPlant-v0', ...)` без церемоний.

Если среда — вклад в сам TensorAeroSpace, откройте PR с:

- функцией динамики в `tensoraerospace/aerospacemodel/<your_model>/`,
- классом среды в `tensoraerospace/envs/<your_model>/`,
- регистрацией в `tensoraerospace/envs/__init__.py`,
- 1–2 примерами-ноутбуками в `example/`,
- страницей `docs/{en,ru}/model/<your_model>.md`, прописанной в `mkdocs.yml`.

## Подводные камни

- **Забытый `reshape(1, -1)` у reference.** Даже для одного канала среда ожидает 2-D. Если получите `IndexError: too many indices for array` — проверьте это.
- **Несоответствие единиц модели и агента.** Если динамика в радианах, а reference_signal в градусах (`output_rad=False`), ошибка слежения взорвётся. Будьте явны в докстрингах.
- **Слишком большой шаг интегратора.** Euler с `dt = 0.01` нормален для ограниченной по полосе аэрокосмики; переключайтесь на `rk4` для агрессивной динамики привода (короткие постоянные времени) или уменьшайте `dt` до 0.005.
- **Недетерминированный `reset`.** Если вы рандомизируете `initial_state` в `reset()`, принимайте и используйте аргумент `seed` для воспроизводимости.

## Куда дальше

- **[Рецепт 02 — Устройство среды](02_env_anatomy.md)** — контракт, которому должна удовлетворять новая среда.
- **[Рецепт 05 — Обучение deep-RL от и до](05_deep_rl_training.md)** — обучить RL-агента на своей среде.
- **[Рецепт 06 — Онлайн-адаптивные агенты](06_online_adaptive.md)** — развернуть IHDP / iADP / AA-INDI на своей среде.
