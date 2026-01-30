# Action-Dependent Heuristic Dynamic Programming (ADHDP) — каноничное описание

Этот файл — краткая шпаргалка по **ADHDP** в смысле статьи Prokhorov & Wunsch (1997) “Adaptive Critic Designs” и нашей реализации `tensoraerospace.agent.ADHDP`.

Важно: в старой версии этого документа были смешаны **HDP** (критик \(J(x)\)) и **model-based градиенты** (ближе к DHP/ADDHP). Для **ADHDP** по статье ключевое отличие — **action-dependent critic** \(J(x,u)\) и **прямая связь actor→critic** (без обязательной модели).

---

## 1) Архитектура ADHDP (по статье)

ADHDP состоит из двух обучаемых сетей и реального plant/environment:

- **Critic**: аппроксимирует \(J(x,u)\) (cost-to-go / utility-to-go)  
  - вход: \([x;u]\)  
  - выход: скаляр \(J\)
- **Actor**: аппроксимирует \(u=\pi(x)\)  
  - вход: \(x\)  
  - выход: \(u\)
- **Plant / Environment**: выдаёт переход \(x_{t+1}\) и локальную стоимость \(U(t)\) (utility).

> В статье: HDP использует model-network для связи actor↔critic, но **ADHDP** — это вариант, где actor подключён к critic напрямую (можно ждать следующего шага, либо модель не подходит).

---

## 2) Bellman-уравнение для ADHDP

Для дискретного времени:

\[
J(x_t,u_t) = U(t) + \gamma J(x_{t+1}, u_{t+1}), \quad u_{t+1}=\pi(x_{t+1})
\]

где:
- \(U(t)\) — локальная стоимость (utility)  
- \(\gamma \in (0,1]\) — discount factor

---

## 3) Обучение (online TD + minimization)

### 3.1 Critic update (TD, semi-gradient)

На каждом шаге строим TD-target:

\[
y_t = U(t) + \gamma\, J(x_{t+1}, \pi(x_{t+1}))
\]

и минимизируем:

\[
L_c = \tfrac{1}{2}\left(J(x_t,u_t) - y_t\right)^2
\]

Как в статье, используем **semi-gradient**: целевой \(y_t\) считается константой относительно весов критика (не дифференцируем \(J(x_{t+1},\pi(x_{t+1}))\) по весам критика внутри таргета).

### 3.2 Actor update (минимизация \(J\))

Actor обучается минимизировать “стоимость” по critic:

\[
L_a = J(x_t,\pi(x_t))
\]

Градиент идёт через critic по входу \(u\) (direct path actor→critic), что соответствует Fig. 1(b) из статьи (“ошибка 1” для минимизации \(J\)).

---

## 4) Training procedure (Section III из статьи): alternating cycles

Статья рекомендует **чередование двух циклов**:

- **critic cycle**: обновляем critic, actor держим фиксированным
- **action cycle**: обновляем actor, critic держим фиксированным

Это напоминает **policy iteration**: во время actor-cycle reward/utility может быть “неровным”, пока critic не догонит новую политику.

Также статья подчёркивает, что система должна оставаться стабильной во время адаптации; рекомендуют начинать обучение критика с actor, который уже является стабилизирующим контроллером.

---

## 5) Практика в TensorAeroSpace (как это отражено в нашей реализации)

В `tensoraerospace.agent.ADHDP` реализовано:

- **Онлайн TD без replay/target** (канонично для ADHDP из статьи).
- **Опциональный warm-start** (имитация baseline-контроллера) + **critic warmup**.
- **Alternating cycles** через параметры `critic_cycle_episodes` / `action_cycle_episodes`.
- **Per-step update multipliers**: `critic_updates_per_step` / `actor_updates_per_step` (аналог “epochs per step” из некоторых практических реализаций).
- **Utility vs shaped reward**: предпочтительно обучаться на `cost_total` (это ближе к \(U(t)\) в статье).

Опция **residual-policy**:

\[
u = u_{baseline}(obs) + \alpha\, u_{actor}(obs)
\]

— это практический стабилизатор (не обязателен “по канону”), но часто помогает избежать saturation/развала, пока critic неточен.

