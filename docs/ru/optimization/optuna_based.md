# Поиск гиперпараметров агента

Инструменты автоматической оптимизации гиперпараметров для алгоритмов управления. Поддерживаются Optuna и Ray Tune.

## Быстрый старт (Optuna)

```python
import numpy as np
from tensoraerospace.optimization import HyperParamOptimizationOptuna

def objective(trial):
    # Пример: минимизируем квадратичную ошибку модели с гиперпараметрами
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    # ... обучите/оцените модель ...
    score = np.random.rand()  # замените на реальную метрику качества
    return score

opt = HyperParamOptimizationOptuna(direction="minimize")
opt.run_optimization(objective, n_trials=20)

best = opt.get_best_param()
print("Лучшие параметры:", best)
opt.plot_parms(figsize=(12, 4))
```

## Классы

=== "Optuna"

    ::: tensoraerospace.optimization.HyperParamOptimizationOptuna

=== "Ray Tune"

    ::: tensoraerospace.optimization.HyperParamOptimizationRay

## Примечания

- `direction`: выберите `"minimize"` или `"maximize"` для целевой метрики.
- Для Ray Tune используйте `run_optimization(func, param_space, tune_config=...)` и далее обрабатывайте `self.results` (см. Ray Tune docs).
# Optimization

**В разработке** - данная страница будет содержать документацию по оптимизации.

<!-- TODO: Конвертировать из optimization/optuna_based.rst -->
