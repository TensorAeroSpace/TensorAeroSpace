# Action-Dependent Heuristic Dynamic Programming (ADHDP) — Canonical Description

This document is a concise reference for **ADHDP** as described in Prokhorov & Wunsch (1997) "Adaptive Critic Designs" and our implementation `tensoraerospace.agent.ADHDP`.

Important: an earlier version of this document mixed up **HDP** (critic \(J(x)\)) and **model-based gradients** (closer to DHP/ADDHP). For **ADHDP** per the paper, the key distinction is the **action-dependent critic** \(J(x,u)\) and the **direct actor→critic connection** (no model required).

---

## 1) ADHDP Architecture (per the paper)

ADHDP consists of two trainable networks and a real plant/environment:

- **Critic**: approximates \(J(x,u)\) (cost-to-go / utility-to-go)  
  - input: \([x;u]\)  
  - output: scalar \(J\)
- **Actor**: approximates \(u=\pi(x)\)  
  - input: \(x\)  
  - output: \(u\)
- **Plant / Environment**: produces transition \(x_{t+1}\) and local cost \(U(t)\) (utility).

> In the paper: HDP uses a model-network to link actor↔critic, but **ADHDP** is the variant where the actor connects to the critic directly (either waiting for the next step, or when a model is unavailable).

---

## 2) Bellman Equation for ADHDP

For discrete time:

\[
J(x_t,u_t) = U(t) + \gamma J(x_{t+1}, u_{t+1}), \quad u_{t+1}=\pi(x_{t+1})
\]

where:
- \(U(t)\) — local cost (utility)  
- \(\gamma \in (0,1]\) — discount factor

---

## 3) Training (Online TD + Minimization)

### 3.1 Critic Update (TD, semi-gradient)

At each step, construct the TD target:

\[
y_t = U(t) + \gamma\, J(x_{t+1}, \pi(x_{t+1}))
\]

and minimize:

\[
L_c = \tfrac{1}{2}\left(J(x_t,u_t) - y_t\right)^2
\]

As in the paper, we use **semi-gradient**: the target \(y_t\) is treated as a constant with respect to critic weights (we do not differentiate \(J(x_{t+1},\pi(x_{t+1}))\) through the critic weights inside the target).

### 3.2 Actor Update (minimizing \(J\))

The actor is trained to minimize the "cost" as estimated by the critic:

\[
L_a = J(x_t,\pi(x_t))
\]

The gradient flows through the critic with respect to input \(u\) (direct path actor→critic), corresponding to Fig. 1(b) in the paper ("error 1" for minimizing \(J\)).

---

## 4) Training Procedure (Section III of the paper): Alternating Cycles

The paper recommends **alternating two cycles**:

- **Critic cycle**: update the critic, keep the actor fixed
- **Action cycle**: update the actor, keep the critic fixed

This resembles **policy iteration**: during the actor cycle, reward/utility may be "uneven" until the critic catches up with the new policy.

The paper also emphasizes that the system must remain stable during adaptation; they recommend starting critic training with an actor that already serves as a stabilizing controller.

---

## 5) Practice in TensorAeroSpace (how this is reflected in our implementation)

In `tensoraerospace.agent.ADHDP` the following is implemented:

- **Online TD without replay/target** (canonical for ADHDP per the paper).
- **Optional warm-start** (imitating a baseline controller) + **critic warmup**.
- **Alternating cycles** via `critic_cycle_episodes` / `action_cycle_episodes` parameters.
- **Per-step update multipliers**: `critic_updates_per_step` / `actor_updates_per_step` (analogous to "epochs per step" in some practical implementations).
- **Utility vs shaped reward**: preferably train on `cost_total` (this is closer to \(U(t)\) in the paper).

**Residual-policy** option:

\[
u = u_{baseline}(obs) + \alpha\, u_{actor}(obs)
\]

This is a practical stabilizer (not strictly "canonical") but often helps avoid saturation/divergence while the critic is still inaccurate.
