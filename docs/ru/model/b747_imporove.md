# ImprovedB747Env -- Boeing 747, отслеживание тангажа в продольном канале (нормализованная, RL-совместимая)

На этой странице описана среда ImprovedB747Env, предназначенная для обучения и оценки контроллеров на основе обучения с подкреплением в продольном канале Boeing 747. Среда предоставляет нормализованный интерфейс наблюдений и действий, формированную награду для точного и плавного отслеживания тангажа, а также реалистичные условия завершения.

Эталонная реализация находится в `tensoraerospace/envs/b747.py` (класс `ImprovedB747Env`).

![](../example/agent/sac/img/sac-b747-impoved.jpg)



## Краткое описание

- Пространство наблюдений: 4D, нормализовано в [-1, 1]
- Пространство действий: 1D, нормализовано в [-1, 1] (команда отклонения руля высоты)
- Цель: отслеживание изменяющегося во времени задания по тангажу при сохранении плавности управления
- Награда: квадратичные штрафы за ошибку отслеживания тангажа, рассогласование угловой скорости тангажа с эталонной динамикой, абсолютное воздействие, скорость изменения и рывок воздействия (с масштабированием)
- Завершение: выход за безопасный диапазон тангажа или достижение горизонта по времени

## Наблюдение и действие

Пусть \(\theta\) -- текущий угол тангажа (рад), \(q\) -- угловая скорость тангажа (рад/с), а \(\theta_{ref}(t)\) -- целевой тангаж.

Наблюдение на шаге \(t\) представляет собой 4-мерный вектор:

\[
\mathbf{o}_t = \big[\underbrace{\mathrm{clip}(\tfrac{\theta_{ref}(t) - \theta(t)}{\theta_{max}},-1,1)}_{\text{[0] norm\_pitch\_error}},\; \underbrace{\mathrm{clip}(\tfrac{q(t)}{q_{max}},-1,1)}_{\text{[1] norm\_q}},\; \underbrace{\mathrm{clip}(\tfrac{\theta(t)}{\theta_{max}},-1,1)}_{\text{[2] norm\_\theta}},\; \underbrace{u_{t-1}}_{\text{[3] prev action}}\big]
\]

где

- \(\theta_{max} = 20^{\circ}\) (внутри преобразуется в радианы),
- \(q_{max} = 5^{\circ}/\text{s}\) (внутри преобразуется в рад/с),
- \(u_{t-1}\in[-1,1]\) -- предыдущая нормализованная команда руля высоты.

**Важно**: Вектор наблюдений уже нормализован и готов к использованию. Индекс `[0]` содержит нормализованную ошибку отслеживания тангажа, которая может использоваться непосредственно для пропорционального управления.

Действие -- единственная нормализованная команда \(u_t \in [-1,1]\). Она отображается в физическое отклонение руля высоты (град):

\[
\delta_e^{\circ}(t) = u_t\,\delta_{e,\max}^{\circ}, \quad \delta_{e,\max}^{\circ} = 25^{\circ},
\]

которое преобразуется в радианы перед передачей во внутреннюю модель самолёта.

## Функция награды

Среда использует формированную награду с пятью слагаемыми, стимулирующими точность и плавность при штрафовании чрезмерного воздействия. Награда за шаг:

\[
r_t = -\,s\,\Big(\,w_\theta\,e_\theta^2\; +\; w_q\,e_q^2\; +\; w_u\,u_t^2\; +\; w_{\Delta}\,(\Delta u_t)^2\; +\; w_{\Delta^2}\,(\Delta^2 u_t)^2\Big),
\]

с весами и масштабом по умолчанию:

\[w_\theta=5.0,\quad w_q=0.2,\quad w_{cross}=0.0,\quad w_u=0.003,\quad w_{\Delta}=0.01,\quad w_{\Delta^2}=0.001,\quad s=0.1.\]

Примечание: В реализации присутствует вес перекрёстного члена \(w_{cross}\) (в настоящее время равен 0.0) для возможного использования в будущем при связывании ошибок тангажа и угловой скорости тангажа.

Члены ошибки определяются как

\[
e_\theta = \frac{\theta(t)-\theta_{ref}(t)}{\theta_{max}},\qquad
e_q = \frac{q(t)-\dot\theta_{ref}(t)}{q_{max}},
\]

где \(\dot\theta_{ref}(t)\) -- конечно-разностная производная эталонного тангажа, вычисленная с шагом моделирования \(\Delta t\). Члены гладкости воздействия используют

\[
\Delta u_t = u_t - u_{t-1},\qquad \Delta^2 u_t = u_t - 2 u_{t-1} + u_{t-2}.
\]

Замечания:

- Больший \(w_\theta\) акцентирует точное отслеживание тангажа.
- \(w_q\) демпфирует реакцию относительно наклона эталонного сигнала, снижая перерегулирование и колебания.
- \(w_u, w_{\Delta}, w_{\Delta^2}\) регуляризируют расход энергии и плавность команд, подавляя дребезг.
- Общий масштаб \(s\) удерживает награды в компактном численном диапазоне для стабильности RL.

## Завершение и усечение

- Аварийное завершение: если \(|\theta| > \theta_{max}\), эпизод завершается досрочно и на этом шаге применяется большой штраф (\(-100\)).
- Усечение: эпизод усекается при достижении заданного горизонта (число временных шагов).

## Динамика эпизода

На каждом шаге:

1. Агент выдаёт \(u_t\in[-1,1]\).
2. Среда ограничивает \(u_t\) диапазоном \([-1,1]\), отображает в \(\delta_e^{\circ}\), преобразует в радианы и продвигает внутреннюю модель самолёта.
3. Наблюдение \(\mathbf{o}_{t+1}\) строится с использованием нормализованных сигналов.
4. Награда \(r_t\) вычисляется по формуле выше.
5. Проверяются условия завершения/усечения.

## Пример использования

```python
import numpy as np
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

dt = 0.1
tn = 200
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

# Эталон: плавный синусоидальный тангаж в радианах (амплитуда 1 град)
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps), frequency=0.05, amplitude=np.deg2rad(1.0), vertical_shift=0.0
    ),
    (1, -1),
)

# Начальное состояние: [u, w, q, theta]
initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)

env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
    dt=dt,
)
# Убедитесь, что дискретизация модели совпадает с шагом среды
env.unwrapped.model.discretisation_time = dt

obs, info = env.reset()
done = False
while not done:
    # простое пропорциональное управление по нормализованной ошибке тангажа в [-1, 1]
    u = float(np.clip(2.0 * float(obs[0]), -1.0, 1.0))
    obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
    done = bool(terminated or truncated)
```

### Пример с предобученным агентом SAC

Использование предобученного агента SAC с HuggingFace:

```python
import numpy as np
from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

# Создание среды (короткий пример)
dt = 0.1
tn = 200
tp = generate_time_period(tn=tn, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
reference_signal = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps), frequency=0.05, amplitude=np.deg2rad(1.0), vertical_shift=0.0
    ),
    (1, -1),
)
initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)
env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=len(tp),
    dt=dt,
    initial_elevator_deg=0.0,
    use_initial_action_on_first_step=True,
)
env.unwrapped.model.discretisation_time = dt

# Загрузка агента и запуск одного эпизода
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
obs, info = env.reset()
done = False
ret = 0.0
while not done:
    action = agent.select_action(obs, evaluate=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    ret += float(reward)
print(f"Return: {ret:.2f}")
```

## Замечания по реализации

- Класс отображает простой HUD и (опционально) 2D-спрайт при рендеринге; это не влияет на физику или награду.
- Границы нормализации наблюдений (\(\theta_{max}, q_{max}\)) и пределы руля высоты определены внутри класса и при необходимости могут быть настроены.
- Веса награды доступны как атрибуты экземпляра (`w_pitch`, `w_q`, `w_cross`, `w_action`, `w_smooth`, `w_jerk`, `reward_scale`) и могут быть изменены для соответствия конкретным целям управления.
- Внутреннее представление состояния следует порядку `[u, w, q, theta]` (единицы СИ: м/с, м/с, рад/с, рад), но наблюдения нормализованы в `[-1, 1]`.

## Литература

- `tensoraerospace/envs/b747.py` -- полная реализация среды
- Модель продольной динамики Boeing 747: `tensoraerospace/aerospacemodel/b747.py`
