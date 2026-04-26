# Рецепт 08 — Save / load / публикация в HuggingFace

**Цель.** Сохранить чекпойнт адаптивного критика, перезагрузить бит-в-бит и опубликовать на [Hugging Face Hub](https://huggingface.co/). Пять агентов имеют этот общий контракт: IHDP, IM-GDHP, ET-DHP, AA-INDI, iADP.

**Связано.** [Рецепт 06](06_online_adaptive.md) — жизненный цикл агента, в который вписывается чекпойнтинг.

## Контракт

Каждый совместимый агент имеет четыре метода:

| Метод | Назначение |
|---|---|
| `agent.save(path)` | Записать директорию с датой под `path` со всем необходимым для продолжения. Возвращает абсолютный путь. |
| `AgentClass.from_pretrained(loc, access_token=None, version=None)` | Загрузить из локальной директории **или** из `namespace/repo` на Hub. |
| `agent.publish_to_hub(repo_name, folder_path, access_token=None)` | Загрузить уже сохранённую папку на Hub. |
| `agent.get_param_env()` | Построить JSON-сериализуемый конфиг-dict для `save()`. |

Без гимнастики с наследованием. Одинаковы у всех пяти агентов.

## Минимальный round-trip

```python
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

agent = IADPAgent(n_state=1, n_control=1, config=IADPConfig(seed=0))
# ... прогоните несколько шагов среды, чтобы было нетривиальное состояние ...

# 1. Локальное сохранение
run_dir = agent.save('./checkpoints')     # './checkpoints/Apr19_12-34-56_IADPAgent'

# 2. Перезагрузка из той же директории
restored = IADPAgent.from_pretrained(run_dir)

# 3. Следующий predict() даёт бит-идентичный результат
np.testing.assert_allclose(agent.predict(x, ref, k), restored.predict(x, ref, k), atol=1e-12)
```

## Что лежит на диске

Сохранённая директория содержит:

| Файл | Содержимое |
|---|---|
| `config.json` | Полный dataclass-конфиг + аргументы конструктора; массивы (Q, R, G_init, …) как списки. |
| `rls.npz` / `vff_rls.npz` | Матрица параметров RLS `theta`, ковариация, счётчик обновлений, последняя невязка. |
| `value.npz` | Ядерная матрица `P̃` (iADP), веса критика (IHDP/IMGDHP/ET-DHP) и т.д. |
| `weights.npz` | Активные Q, R (с подставленными дефолтами). |
| `loop_state.npz` | Rolling state — `X_prev`, `delta_prev`, состояние интегратора, счётчик шагов. |
| `window.npz` | Буфер переходов policy-evaluation (iADP). |
| `deriv_state.npz`, `bias_state.npz` | Состояния сенсорных фильтров (AA-INDI). |

**Mid-episode сохранения бит-идентично восстанавливаются** — гарантия, проверенная тестами библиотеки. Ключевое отличие от наивного `pickle` агента: контракт сохраняет *только нужное*, перезагрузка восстанавливается из `config.json`.

## Round-trip с HuggingFace Hub

### Загрузка

```python
run_dir = agent.save('./checkpoints')

agent.publish_to_hub(
    repo_name='your-username/iadp-f16-v1',
    folder_path=run_dir,
    access_token='hf_...',        # из https://huggingface.co/settings/tokens
)
```

Если `repo_name` не существует на Hub, он создаётся. Загружается весь `run_dir` как есть — добавьте `README.md` с контекстом перед вызовом `publish_to_hub`, если хотите нормальную model card.

### Скачивание

```python
restored = IADPAgent.from_pretrained(
    repo_name='your-username/iadp-f16-v1',
    access_token='hf_...',        # нужен только для приватных repo
    version='main',               # опциональная ветка / tag / commit
)
```

Внутри `from_pretrained` вызывает `huggingface_hub.snapshot_download(repo_name)`, когда путь — не локальная директория.

## Минимальная model card

Добавьте `README.md` в `run_dir` перед `publish_to_hub`:

```markdown
---
tags:
  - tensoraerospace
  - flight-control
  - iadp
library_name: tensoraerospace
---
# iADP на нелинейной F-16 (v1)

Агент отслеживания угловой скорости тангажа, обучен на `NonlinearLongitudinalF16-v0`
при `dt = 0.01` с, с DARE-based `P_init` и `policy_eval_blend = 0.1` (см.
[рецепт cookbook](https://...)).

## Назначение
Онлайн-адаптивное слежение ω_z в точке трима F-16.

## Гиперпараметры
- Q = 30 000, R = 0.1, γ = 0.9
- γ_RLS = 0.9999, φ_init = 1.0
- policy_eval_blend = 0.1, policy_eval_every = 5

## Воспроизвести
`IADPAgent.from_pretrained('your-username/iadp-f16-v1')`
```

Любой YAML frontmatter рендерится Hub'ом нативно.

## Подводные камни

- **Путь, начинающийся с `./`, `../`, `/`, `~`, трактуется как локальный.** Если локального пути нет, `from_pretrained` кидает `FileNotFoundError` вместо попытки Hub-скачивания.
- **Массивы в `config.json` должны быть JSON-сериализуемы.** Библиотека конвертирует NumPy-массивы в списки внутри `get_param_env()`; если наследуете — сохраните это поведение.
- **Старые чекпойнты без `loop_state.npz`** получат свежесконструированное нулевое состояние. То есть ep-level перезагрузка безопасна между minor-версиями; mid-episode требует ту же minor-версию, что писала чекпойнт.
- **Загрузка больших window-буферов.** У агентов с буфером переходов (iADP) `window.npz` растёт с `policy_eval_window`. При `policy_eval_window=300` на 2-D augmented state это ~4 КБ — ничего. При `policy_eval_window=10 000` на 6-DoF стоит проверить.

## Живой пример

Ноутбук [iADP на нелинейной F-16](../example/agent/iadp/example_iadp_nonlinear.md) тестирует полный round-trip в рамках своих тестов.

## Куда дальше

- **[Рецепт 09 — Отказоустойчивость](09_fault_tolerance.md)** — сохранение агента в условиях отказа и безопасная перезагрузка.
- **[Рецепт 10 — Добавить свой объект управления](10_custom_plant.md)** — расширение контракта персистентности на свои агенты.
