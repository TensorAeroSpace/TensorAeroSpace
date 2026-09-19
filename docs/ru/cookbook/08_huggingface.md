# Рецепт 08 — Сохранение, продолжение и публикация адаптивных регуляторов

**Цель:** сохранить агента с накопленной историей обучения, загрузить его и
проверить совпадение следующих 50 физических переходов. Затем подготовить
checkpoint для необязательной загрузки на Hugging Face. Выполняйте локальные
блоки Python по порядку с установленной текущей версией `tensoraerospace`. Сетевые операции автоматически
не запускаются.

Для полезного checkpoint нужно понимать его состав. Состояния регулятора и
симулятора хранятся отдельно, а AA-INDI и iADP используют разные форматы файлов.

## 1. Определите границу сохранения

| Компонент | Сохраняет агент? | Что должен сохранить эксперимент |
|---|---|---|
| Обученная модель / критик | Да, для соответствующего алгоритма | Ту же реализацию и совместимую конфигурацию |
| История регулятора и фильтры | Включены в текущие checkpoint iADP и AA-INDI | Следующее измерение и порядок вызовов |
| Состояние самолёта / приводов / двигателей | Нет | Физические состояния, состояния приводов и параметры модели |
| Часы симуляции и выполненные события | Нет | Время/индекс и уже применённые отказы объекта |
| Генератор задания / случайных чисел | Внешние генераторы не сохраняются | Их состояния или полные воспроизводимые последовательности |

`reset()` не восстанавливает сохранение. У этих адаптивных агентов он очищает
историю цикла, сохраняя обученные параметры. Для независимого номинального прогона
создавайте нового агента; для продолжения сохранённого — вызывайте `from_pretrained`.

## 2. Создайте checkpoint iADP с реальной историей обучения

Небольшой детерминированный объект описывается `x_next = a*x + b*u`. Номинальная
инициализация через DARE позволяет сосредоточиться на сохранении; это не регулятор
самолёта.


```python
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_are
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

dt = 0.01
a = np.exp(-2 * dt)
b = -np.expm1(-2 * dt) / 2
F = np.diag([a, 1.0])
G = np.array([[b], [0.0]])
gamma = 0.9
R = np.array([[0.01]])
Q_aug = np.array([[1.0, -1.0], [-1.0, 1.0]])
P = solve_discrete_are(np.sqrt(gamma) * F, np.sqrt(gamma) * G, Q_aug, R)
agent = IADPAgent(1, 1, IADPConfig(
    dt=dt, Q=np.eye(1), R=R, gamma=gamma,
    F_init=F, G_init=G, P_init=P,
    learning_mode="continuous", policy_eval_window=100,
    policy_eval_min_samples=100, policy_eval_every=20,
    u_magnitude_limit=0.5, u_rate_limit=2.0,
))
x, reference = np.zeros(1), np.array([0.05])
for k in range(200):
    u = agent.predict(x, reference, k)
    x = a * x + b * u
    agent.learn(x, reference, k, applied_action=u)
```


Теперь регулятор прошёл 200 переходов. Сохраните его сразу после `learn`, чтобы
было однозначно понятно, что ожидающих применения команд нет.


```python
run_dir = Path(agent.save("./checkpoints/recipe08-iadp"))
# The controller checkpoint does not contain the external plant.
(run_dir / "simulation.json").write_text(json.dumps({
    "state": x.tolist(), "next_step": 200,
    "reference": reference.tolist(), "dt": dt, "a": a, "b": b,
}, indent=2))
print("Saved files:", sorted(path.name for path in run_dir.iterdir()))

restored = IADPAgent.from_pretrained(str(run_dir))
simulation = json.loads((run_dir / "simulation.json").read_text())
x_restored = np.array(simulation["state"])
reference_restored = np.array(simulation["reference"])
for k in range(simulation["next_step"], simulation["next_step"] + 50):
    u = agent.predict(x, reference, k)
    u_restored = restored.predict(x_restored, reference_restored, k)
    np.testing.assert_array_equal(u, u_restored)
    x = a * x + b * u
    x_restored = simulation["a"] * x_restored + simulation["b"] * u_restored
    agent.learn(x, reference, k, applied_action=u)
    restored.learn(x_restored, reference_restored, k, applied_action=u_restored)
    np.testing.assert_array_equal(x, x_restored)
    np.testing.assert_array_equal(agent.P, restored.P)
    np.testing.assert_array_equal(agent.rls.theta, restored.rls.theta)
print("50 continued transitions match exactly in this process")
```


Проверяются действия, состояния объекта, коэффициенты RLS и матрицы критика
на следующих 50 переходах, включая обучение. Точное равенство здесь относится
к детерминированному продолжению в том же программном окружении. Другая реализация
BLAS или версия библиотек может изменить результаты вычислений с плавающей точкой.

`simulation.json` принадлежит обвязке примера. Для полного самолёта одного
скалярного состояния недостаточно: нужно восстановить окружение с теми же
параметрами, временем, уже применёнными отказами, приводами и генераторами датчиков.

## 3. Разберитесь в файлах каждого агента

| Агент | Текущие файлы и содержимое |
|---|---|
| iADP | `config.json`: конструктор, настройки и матрицы выходов; `rls.npz`: инкрементальная модель, ковариация и счётчики; `value.npz`: критик; `weights.npz`: веса стоимости; `loop_state.npz`: история команд и переходов; `window.npz`: выборка критика. |
| AA-INDI | `paper_aaindi.json`: геометрия и настройки, оценки моментов, ковариационное состояние OTSEKF, состояние HOSM, отфильтрованные измерения и ожидающая команда. |

Используйте путь, возвращённый `save`, вместо ручного составления имени каталога.
Путь может быть относительным, если относительным был родительский каталог.
Для независимых прогонов задавайте разные родительские каталоги: метка времени
в имени имеет точность до секунды.

Для IHDP, IM-GDHP и ET-DHP ориентируйтесь на разделы сохранения
в [документации IHDP](../agent/ihdp.md), [IM-GDHP](../agent/imgdhp.md)
и [ET-DHP](../agent/et_dhp.md). Нельзя предполагать одинаковый набор файлов
и дополнительных аргументов конструктора у всех алгоритмов.

## 4. Сохраните AA-INDI через локальный интерфейс

Загрузчик AA-INDI принимает **локальный каталог**. Этот самостоятельный пример
использует номинальную инициализацию B747 и корректный пакет датчиков в СИ;
проверяется команда на сохранённой метке времени. Полный цикл полёта приведён
в [рецепте 14](14_aaindi.md).


```python
from tensoraerospace.agent.aa_indi import AAINDIAgent
from tensoraerospace.benchmark import B747EngineFailureBenchmark

from tensoraerospace.agent.aa_indi import FlightMeasurement

benchmark = B747EngineFailureBenchmark(dt=0.02)
aa = benchmark.make_aaindi()
trim = benchmark.nominal_trim()
state = trim.to_state()
action = np.array([trim.elevator_rad, 0.0, 0.0, trim.throttle])
sample = FlightMeasurement.from_model(benchmark.nominal_model(), applied_action=action, surface_indices=(1, 2))
rate_command = np.array([0.001, 0.0, 0.0])
aa.predict(sample, rate_command)
aa_dir = aa.save("./checkpoints/recipe08-aaindi")
aa_restored = AAINDIAgent.from_pretrained(aa_dir)
np.testing.assert_array_equal(
    aa.predict(sample, rate_command), aa_restored.predict(sample, rate_command),
)
print("AA-INDI local checkpoint:", aa_dir)
```


При сохранении между `predict` и `learn` ожидающая команда сохраняется.
Продолжайте с её однократного применения и вызова `learn` с новым пакетом датчиков.
После сохранения за `learn` продолжайте со следующего `predict` на том же пакете.
Не создавайте другое наблюдение с уже использованной меткой времени.

## 5. Добавьте контекст для воспроизведения

Перед публикацией поместите рядом с checkpoint файл `README.md`. Укажите:

- Коммит репозитория, версии Python/зависимостей и класс агента.
- Конфигурацию самолёта, балансировку, шаг и интегратор.
- Порядок состояний, переводы СИ/US, абсолютные команды или отклонения от балансировки, ограничения приводов.
- Задание, seed, коэффициенты регулятора и номинальную инициализацию.
- Положение сохранения относительно `learn` и способ восстановления объекта.
- Метрики исправного и аварийного прогонов с окнами оценки, незавершённые и расходящиеся прогоны.

Для B747 добавьте ссылку на [протокол сравнения](../comparison/aaindi_vs_pid_lqr_lqi_b747.md).
Сохранённая модель сама по себе не описывает допущения оценки и не доказывает
устойчивость к отказам.

## 6. Явно запустите публикацию или скачивание

Следующие функции используют установленный пакет `huggingface_hub`. Их можно
определить локально; примеры вызовов оставлены в комментариях для момента,
когда понадобится публикация или скачивание. Передавайте `HF_TOKEN` через
окружение, не записывая токен в ноутбук.


```python
import os
from huggingface_hub import HfApi, snapshot_download

def upload_checkpoint(folder, repo_id):
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder))

def download_aaindi(repo_id, revision):
    folder = snapshot_download(
        repo_id=repo_id, revision=revision, token=os.environ.get("HF_TOKEN"),
    )
    return AAINDIAgent.from_pretrained(folder)

# Explicit optional network operations, after replacing the repository and revision:
# upload_checkpoint(aa_dir, "your-username/aaindi-b747")
# downloaded = download_aaindi("your-username/aaindi-b747", "COMMIT_SHA")
# downloaded_iadp = IADPAgent.from_pretrained(
#     "your-username/iadp-example", version="COMMIT_SHA",
#     access_token=os.environ.get("HF_TOKEN"),
# )
```


`create_repo` явно создаёт репозиторий при необходимости. Новый репозиторий
в этом примере приватный. Для воспроизводимого скачивания указывайте ревизию
коммита. Текущий iADP также имеет `publish_to_hub`, загружающий файлы в существующий
репозиторий, и `from_pretrained` с поддержкой Hub. У AA-INDI загрузчик локальный,
метода `publish_to_hub` нет: сначала скачайте файлы, затем загрузите каталог.

## Типичные проблемы

| Симптом | Возможная причина и проверка |
|---|---|
| Первая команда после загрузки отличается | Различаются измерение объекта, задание, ожидающий переход или время. Сначала сравните их. |
| Одна команда совпадает, затем начинается расхождение | Восстановите также симулятор, приводы и RNG; сравнивайте несколько шагов обучения. |
| Нет `paper_aaindi.json` | Каталог не является сохранением текущей реализации AA-INDI. |
| Старая конфигурация iADP отвергается | Удалённые настройки регуляризации и смешивания критика несовместимы с текущим законом обновления; создайте checkpoint заново. |
| AA-INDI не загружает `username/repo` | Сначала скачайте репозиторий в локальный каталог. |
| При публикации репозиторий не найден | Явно создайте его и проверьте доступ. |

В старых checkpoint AA-INDI только по угловой скорости нет физической геометрии
и независимого навигационного состояния. Их загрузку нельзя считать эквивалентным
продолжением. Храните ревизию исходников с экспериментом и пересоздавайте
несовместимые сохранения.

**Далее:** [Рецепт 09 — Отказоустойчивость](09_fault_tolerance.md) ·
[Рецепт 14 — AA-INDI на B737](14_aaindi.md).
