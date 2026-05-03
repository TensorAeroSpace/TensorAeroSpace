# Mathematics of damage modeling

This document collects every formula used by the damage subsystem in one
place: from notation and state evolution to the final form of the 6-DoF
ODEs with damage applied. Abbreviations: **CG** — centre of gravity,
**ODE** — ordinary differential equation, **MAC** — Mean Aerodynamic Chord.

- Overview and quick start — [Aircraft damage modeling](aircraft-damage-modeling.md).
- Implementation and worked examples — [Implementation and examples](aircraft-damage-modeling-code.md).
- The formulas live in
  `tensoraerospace/aerospacemodel/f16/nonlinear/damage/recompute.py` and
  `aero_corrections.py`.

## Notation and units

All vectors are in the body-fixed frame: $x$ forward, $y$ right, $z$ down.

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\mathcal{S} = \{s_1, \ldots, s_N\}$ | set of sections ($N=13$ for the F-16) | — |
| $f_s \in [0, 1]$ | current loss fraction of section $s$ (`section_loss`) | — |
| $m_s$, $A_s$, $c_s$ | nominal mass, area, chord of a section | kg, m², m |
| $\mathbf{r}_s = (x_s, y_s, z_s)^\top$ | body-frame position of the section CG | m |
| $\mathbf{I}^{loc}_s = (I^{loc}_{xx,s},\,I^{loc}_{yy,s},\,I^{loc}_{zz,s},\,I^{loc}_{xy,s})$ | local inertia tensor about the section CG | kg·m² |
| $C_{l\alpha,s}$ | per-section lift-curve slope contribution | 1/rad |
| $C_{D0,s}$ | per-section parasitic drag contribution | — |
| $\alpha$, $\beta$ | angle of attack and sideslip | rad |
| $S_{base}$, $b_{base}$, $\bar{c}_{base}$ | base (un-damaged) area, span, MAC | m², m, m |
| $\Delta m_{struct}$, $\Delta \mathbf{r}_{struct}$, $\Delta \mathbf{J}_{struct}$ | structural deltas (dropped stores, ice) | kg, m, kg·m² |

!!! note "Convention for moment coefficients vs Mach number"

    In this document **$M_x, M_y, M_z$** are the dimensionless rolling,
    pitching, and yawing moment coefficients (the same as $C_l, C_m, C_n$
    in the Western convention; in code: `mx, my, mz`). The scalar **$M$**
    without a subscript is the Mach number. To avoid confusion, Mach
    appears only as a parameter of the base lookup coefficients
    $C_y^{base}(\alpha, M, \ldots)$ and is not given its own section.

## Time evolution of the damage state

The damage state $\mathbf{D}(t)$ is a piecewise-constant function of
time, updated only at event-trigger instants. Let
$\mathcal{E}_k = \{e \in \mathcal{P} \cup \mathcal{P}_{inj} : t_{k-1} < t_e \le t_k\}$
be the set of events from the scheduled profile and one-shot injected
events that fall in the integrator window $(t_{k-1}, t_k]$. Then

$$
\mathbf{D}(t) \;=\; \mathbf{D}(t_{k-1}) \;\oplus\; \bigoplus_{e \in \mathcal{E}_k} \Phi_e, \qquad t \in [t_{k-1}, t_k),
$$

where $\Phi_e$ is the operator that applies event $e$ to the state
(mutating `section_loss`, `control_failures`, `engine`, or `structural`),
and $\oplus$ is the binary composition operator that associates updates
in insertion order. If $\mathcal{E}_k = \emptyset$, then
$\mathbf{D}(t) = \mathbf{D}(t_{k-1})$ and **no parameter recompute is
performed** — an important optimisation: this guarantees bit-identity
with the no-damage baseline.

## Effective mass and centre of gravity

The effective mass of each section drops linearly with its loss fraction:

$$
m^{eff}_s \;=\; m_s\,(1 - f_s).
$$

Total aircraft mass with the structural correction:

$$
m \;=\; \sum_{s \in \mathcal{S}} m^{eff}_s \;+\; \Delta m_{struct}.
$$

The CG is the mass-weighted average of surviving sections plus the
structural shift:

$$
\mathbf{r}_{cg} \;=\; \frac{1}{\displaystyle\sum_s m^{eff}_s}\,\sum_{s \in \mathcal{S}} m^{eff}_s\,\mathbf{r}_s \;+\; \Delta\mathbf{r}_{struct}.
$$

Symmetric loss (e.g., both wingtips) keeps $y_{cg} = 0$ because the
$\pm y_s$ terms cancel. Asymmetric loss produces a non-zero $y_{cg}$ —
the driver of the rolling-moment imbalance.

## Aerodynamic aggregates

The effective wing area sums only `wing` sections:

$$
S \;=\; \sum_{s \in \mathcal{W}} A_s\,(1 - f_s), \qquad \mathcal{W} = \{s \in \mathcal{S}:\,\text{type}_s = \text{wing}\}.
$$

The effective span is the sum of distances to the outermost surviving
point on each side (handles asymmetric loss where one side's wing is
shorter):

$$
b \;=\; \max_{\substack{s \in \mathcal{W}\\ \text{side}_s = L}} |y_s|\cdot\mathbb{1}[f_s < 1] \;+\; \max_{\substack{s \in \mathcal{W}\\ \text{side}_s = R}} |y_s|\cdot\mathbb{1}[f_s < 1].
$$

If a side has no surviving sections ($f_s = 1$ for all of them), the
corresponding $\max$ is over the empty set and is treated as 0 (full
half-wing loss).

The Mean Aerodynamic Chord (MAC) is the area-weighted average chord over
the surviving area:

$$
\bar{c} \;=\; \frac{1}{S}\,\sum_{s \in \mathcal{W}} c_s\,A_s\,(1 - f_s),
$$

with $\bar{c} = 0$ when $S = 0$ (full wing loss).

## Inertia tensor via Huygens-Steiner

For each section we apply the parallel-axis theorem, anchoring the local
inertia to the **current** aircraft CG $\mathbf{r}_{cg}$. Let
$\mathbf{r}'_s = \mathbf{r}_s - \mathbf{r}_{cg} = (r_x, r_y, r_z)_s$ be
the section CG offset relative to the aircraft CG. Local moments scale
by $(1 - f_s)$ under the uniform-density assumption (a 30 % mass loss
removes 30 % of the section's local $I^{loc}$):

$$
J_{xx} \;=\; \sum_{s} \Bigl[ I^{loc}_{xx,s}(1-f_s) \;+\; m^{eff}_s\,(r_{y,s}^2 + r_{z,s}^2)\Bigr] \;+\; \Delta J^{struct}_x,
$$

$$
J_{yy} \;=\; \sum_{s} \Bigl[ I^{loc}_{yy,s}(1-f_s) \;+\; m^{eff}_s\,(r_{x,s}^2 + r_{z,s}^2)\Bigr] \;+\; \Delta J^{struct}_y,
$$

$$
J_{zz} \;=\; \sum_{s} \Bigl[ I^{loc}_{zz,s}(1-f_s) \;+\; m^{eff}_s\,(r_{x,s}^2 + r_{y,s}^2)\Bigr] \;+\; \Delta J^{struct}_z,
$$

$$
J_{xy} \;=\; \sum_{s} \Bigl[ I^{loc}_{xy,s}(1-f_s) \;+\; m^{eff}_s\,r_{x,s}\,r_{y,s}\Bigr] \;+\; \Delta J^{struct}_{xy}.
$$

The off-diagonal sign `+m·rx·ry` matches the integral definition
$J_{xy} = \int x\,y\,dm$ (not $-\int x\,y\,dm$): this is consistent with
`F16AngularParameters.Jxy = 1331.4` and with the structure of
`f16_ode_6dof`, where **`Jxy` (not `Jxz`) is the active off-diagonal** —
a deliberate departure from the standard Western aerospace convention.

## Strip-theory aerodynamic deltas

After each event triggers, the F-16 base coefficient lookup tables
($C_y^{base}, C_x^{base}, \ldots$, functions of $\alpha, \beta, M, \delta_{ail}, \ldots$)
**are not modified**. Instead the subsystem produces additive
dimensionless deltas, normalised by the **base** (pre-damage) geometry
$S_{base}$ and $b_{base} = 2\,\max_{s \in \mathcal{W}} |y_s|$.

**Lift.** Each damaged wing section loses a fraction of its
$C_{l\alpha,s}\,\alpha$ contribution; the section weight is its
area-to-base-area ratio:

$$
\boxed{\;\Delta C_y(\alpha) \;=\; -\sum_{s \in \mathcal{W}} C_{l\alpha,s}\,\alpha\,f_s\,\frac{A_s}{S_{base}}\;}
$$

The minus sign: damage **removes** lift.

**Drag with the "jagged-edge" model.** Drag includes two effects:
(1) losing a section's $C_{D0,s}$ contribution at full loss reduces
parasitic drag, and (2) a partially destroyed section **adds** extra
drag from its torn edge, peaking at $f = 0.5$ where the cross-section
is most exposed:

$$
\boxed{\;\Delta C_x \;=\; \sum_{s \in \mathcal{S}} \Bigl[-C_{D0,s}\,f_s\Bigr] \;+\; \sum_{s \in \mathcal{W}} k_J\,f_s\,(1 - f_s)\,\frac{A_s}{S_{base}}\;}
$$

where $k_J = 0.05$ is a heuristic calibration coefficient. The parabola
$f(1-f)$ peaks at $0.25$ for $f = 0.5$, matching physical intuition:
intact and fully torn sections have clean flow, half-cut surfaces give
the worst turbulent drag.

**Side force** is dominated by the vertical tail:

$$
\boxed{\;\Delta C_z(\beta) \;=\; -\sum_{s \in \mathcal{V}} k_{vt}\,\beta\,f_s, \qquad k_{vt} = 0.40\ \text{rad}^{-1}\;}
$$

where $\mathcal{V}$ is the vtail-typed sections. $k_{vt}$ is calibrated
so a full vtail loss yields $\partial C_z / \partial\beta \approx 0.40$,
matching the F-16 vertical tail's contribution to side stability.

**Rolling moment** arises only under asymmetry — each section contributes
a lever-arm-weighted term $y_s/b_{base}$:

$$
\boxed{\;\Delta M_x(\alpha) \;=\; -\sum_{s \in \mathcal{W}} C_{l\alpha,s}\,\alpha\,f_s\,\frac{A_s}{S_{base}}\,\frac{y_s}{b_{base}}\;}
$$

Symmetric loss: $y_s$ for the left and right sections has opposite
signs and the terms cancel, giving $\Delta M_x = 0$. Asymmetric loss
(e.g. `left_tip` only): a non-zero net $\Delta M_x$ remains, scaling
with both $\alpha$ and $f$ — this is the roll-imbalance driver in
`WING_STRIKE_LEFT_TIP`.

**Pitching moment** uses the lever arm $x_s$ from the CG to each
section's aerodynamic centre (so the formula includes both wing and
stabilator sections):

$$
\boxed{\;\Delta M_y(\alpha) \;=\; -\sum_{s \in \mathcal{W} \cup \mathcal{T}} C_{l\alpha,s}\,\alpha\,f_s\,\frac{A_s}{S_{base}}\,\frac{x^{arm}_s}{\bar{c}_{base}}\;}
$$

where $\mathcal{T}$ is the stab-typed sections and $x^{arm}_s$ is
`aero_x_arm` (lever arm from the aircraft CG to the section's
aerodynamic centre).

**Yawing moment** — asymmetric drag about $z$:

$$
\boxed{\;\Delta M_z \;=\; \sum_{s \in \mathcal{S}} \delta C_{x,s}\,\frac{y_s}{b_{base}}\;}
$$

where $\delta C_{x,s}$ is the per-section $\Delta C_x$ contribution
(including the jagged-edge term for wing sections).

## Control surface failures

Actuator commands $\mathbf{u}_{cmd}$ are mapped to effective
$\mathbf{u}_{eff}$ **before** the integrator through an element-wise
mapping in `apply_control_failures(u, state)`:

$$
u_{eff,i} \;=\; \begin{cases}
\;u^{jam}_i & \text{if mode = jam}\\
\;\eta_i\,u_{cmd,i} & \text{if mode = efficiency\_loss},\;\eta_i \in [0, 1]\\
\;0 & \text{if mode} \in \{\text{lost}, \text{free\_floating}\}\\
\;u_{cmd,i} & \text{otherwise (healthy)}
\end{cases}
$$

Here $\eta_i$ is the `efficiency` field of the `ControlFailure` object
and $u^{jam}_i$ is `jam_position_rad`. Multiple failures on one index
compose in `state.control_failures` insertion order. For split-stab mode
($\mathbf{u} \in \mathbb{R}^4$): indices `stab_left=0`, `stab_right=1`,
`aileron=2`, `rudder=3`.

## Engine

$$
T_{eff} \;=\; \begin{cases}
\;0 & \text{if hard\_failure}\\
\;\tau\,T_{base} & \text{otherwise},\;\tau \in [0, 1]
\end{cases}
$$

The current angular ODE keeps airspeed constant, so $T_{eff}$ is read
separately by consumers (RL rewards, future dynamics extensions) through
the `effective_thrust(base_thrust, state)` helper in
`damage/propulsion.py`. Full thrust integration in the ODEs is a future
extension.

## Coupling to the equations of motion

At every integrator step the full coefficients are the sum of the
baseline lookup and the current delta:

$$
C_y(t) = C_y^{base}(\alpha, M, \ldots) + \Delta C_y(t),\qquad
M_i(t) = M_i^{base}(\alpha, \beta, \boldsymbol{\delta}, \ldots) + \Delta M_i(t),\;\;i \in \{x,y,z\}.
$$

The 6-DoF body-frame equations of motion with **updated** $m$,
$\mathbf{J}$, $S$, $\bar{c}$, $b$ become (sketch, for illustration):

$$
m\,\dot{\mathbf{V}} + \boldsymbol{\omega} \times m\,\mathbf{V} \;=\; \mathbf{F}_{aero}\bigl(C_x, C_y, C_z;\,S,\,q\bigr) + \mathbf{F}_{thrust}(T_{eff}) + m\,\mathbf{g},
$$

$$
\mathbf{J}\,\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times \mathbf{J}\,\boldsymbol{\omega} \;=\; \mathbf{M}_{aero}\bigl(M_x, M_y, M_z;\,S,\,b,\,\bar{c},\,q\bigr),
$$

with $q = \tfrac{1}{2}\rho V^2$ the dynamic pressure. All vectors and
moments are body-fixed, like the aircraft state. The inertia tensor
$\mathbf{J}$ is now a **function of time**, since `apply_to_params(...)`
rewrites it on every event. This substitution
$\mathbf{J}_{base} \to \mathbf{J}(t)$,
$\mathbf{F}_{aero} \to \mathbf{F}_{aero} + \Delta\mathbf{F}$,
$\mathbf{u}_{cmd} \to \mathbf{u}_{eff}$ is what turns the F-16 from a
**fixed-parameter** plant into a **time-varying-parameter** plant
(piecewise-constant — parameters jump at events, stay frozen between them).

## Computational cycle, summarised

At every integrator step:

1. `damage_manager.update(t_curr, t_prev)` collects $\mathcal{E}_k$.
2. If $\mathcal{E}_k \neq \emptyset$: apply the operators $\Phi_e$, then
   `apply_to_params` recomputes $m, S, b, \bar{c}, \mathbf{r}_{cg}, \mathbf{J}$.
3. `apply_control_failures(u_cmd, state)` → $\mathbf{u}_{eff}$.
4. `f16_ode_6dof` looks up $C^{base}_*$ from tables on $\alpha, \beta, M$
   and adds $\Delta C_*$ via `delta_cy/cx/cz/mx/my/mz(α, β, geo, state)`.
5. The ODE is integrated (Euler / RK4) with the updated $m, \mathbf{J}$
   and full $C_*$.

Recompute cost is $O(N)$ in the number of sections ($N=13$) and runs
**only** in steps where events fire. Other steps read the cached
parameters directly — that is what guarantees bit-identity with the
no-damage baseline.
