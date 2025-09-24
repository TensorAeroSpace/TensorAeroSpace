# Написание скрипта в MATLAB

Этот код MATLAB инициализирует параметры и начальные условия, запускает симуляцию модели в Simulink, извлекает данные результатов и управляет моделями Simulink.

```matlab
clear;

% init parameters
[A, B, C, D] = uav1_data();

init = [0.1 0 0 0];
ref_signal = -0.1;

t_s = 0;
t_e = 1200;
dt = 0.1;

sim_out = sim('uav1_model.slx');

y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
w = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;

bdclose('all');
open('uav1_model.slx');
```

## Что делает скрипт

1. Очистка рабочей среды

```matlab
clear;
```

Очищает все переменные из рабочей области MATLAB, чтобы избежать конфликтов с предыдущими данными.

1. Инициализация параметров модели

```matlab
[A, B, C, D] = uav1_data();
```

Функция `uav1_data` возвращает матрицы системы (A, B, C, D), описывающие динамику в форме состояния.

1. Начальные условия и параметры симуляции

```matlab
init = [0.1 0 0 0];
ref_signal = -0.1;

t_s = 0;
t_e = 1200;
dt = 0.1;
```

- `init` — начальные условия состояния.
- `ref_signal` — целевой сигнал управления.
- `t_s`, `t_e`, `dt` — времена начала/окончания и шаг моделирования.

1. Запуск симуляции

```matlab
sim_out = sim('uav1_model.slx');
```

Запускает симуляцию модели `uav1_model.slx`. Результат сохраняется в `sim_out`.

1. Извлечение данных

```matlab
y = sim_out.get('yout');

u = y.getElement(1).Values.Data;
w = y.getElement(2).Values.Data;
q = y.getElement(3).Values.Data;
theta = y.getElement(4).Values.Data;
t = y.getElement(5).Values.Data;
```

Достаёт из результатов значения управлений, скоростей, угла тангажа и времени.

1. Закрытие/открытие модели

```matlab
bdclose('all');
open('uav1_model.slx');
```

Закрывает все открытые модели и открывает `uav1_model.slx` для просмотра/редактирования.
