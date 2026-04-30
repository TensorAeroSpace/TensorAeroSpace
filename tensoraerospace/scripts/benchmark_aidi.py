"""AIDI benchmark CLI.

Usage::

    python -m tensoraerospace.scripts.benchmark_aidi \\
        --env f16_nonlinear_angular \\
        --baselines frozen \\
        --scenarios nominal,stab_25,stab_lost \\
        --episodes 5 --steps 1500 \\
        --out report.md --csv report.csv

Each (method, scenario) combo runs ``--episodes`` rollouts and the per-axis
RMSE is averaged; the result is written as Markdown and CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import fsolve

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (
    rudder_total_loss,
    stab_efficiency_step,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE


SCENARIOS = {
    "nominal": (lambda: None),
    "stab_50": (lambda: stab_efficiency_step(t_inject=5.0, mu=0.5)),
    "stab_25": (lambda: stab_efficiency_step(t_inject=5.0, mu=0.25)),
    "stab_lost": (lambda: stab_efficiency_step(t_inject=5.0, mu=0.0)),
    "rudder_lost": (lambda: rudder_total_loss(t_inject=5.0)),
}


def _solve_trim() -> tuple[float, float]:
    params = default_parameters()

    def trim_residual(z):
        alpha, stab = z
        x = np.array([alpha, 0.0, stab, 0.0])
        return list(f16_ode_long(x, np.array([stab]), 0.0, params)[:2])

    sol, _info, ier, _msg = fsolve(
        trim_residual, x0=[math.radians(2.0), math.radians(-2.0)],
        full_output=True,
    )
    if ier != 1:
        raise RuntimeError("F-16 trim solver did not converge")
    return float(sol[0]), float(sol[1])


def _build_agent(method: str) -> AIDIAgent:
    if method == "adaptive":
        cfg = AIDIConfig(
            dt=0.01, u_magnitude_limit=math.radians(20.0),
            u_rate_limit=math.radians(60.0),
            rls_lambda_min=0.7, rls_lambda_max=0.999, rls_cov_init=10.0,
            cstar_kp=0.5, cstar_ki=0.2,
            roll_omega_n=1.5, roll_zeta=0.8,
            sideslip_kp=0.5, sideslip_ki=0.05, seed=0,
        )
    elif method == "frozen":
        cfg = AIDIConfig(
            dt=0.01, u_magnitude_limit=math.radians(20.0),
            u_rate_limit=math.radians(60.0),
            rls_lambda_min=0.999, rls_lambda_max=0.9999, rls_cov_init=10.0,
            cstar_kp=0.5, cstar_ki=0.2,
            roll_omega_n=1.5, roll_zeta=0.8,
            sideslip_kp=0.5, sideslip_ki=0.05, seed=0,
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return AIDIAgent(
        n_state=3, n_control=3,
        onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
        config=cfg,
    )


def _run_episode(
    method: str, scenario_name: str, n_steps: int,
    alpha_trim: float, stab_trim: float,
) -> dict[str, float]:
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    agent = _build_agent(method)
    profile = SCENARIOS[scenario_name]()
    initial_state = np.zeros(14)
    initial_state[0] = alpha_trim; initial_state[8] = stab_trim
    env = NonlinearAngularF16(
        initial_state=initial_state, number_time_steps=n_steps + 2,
        dt=0.01, integrator="rk4", airspeed=200.0,
        damage_profile=profile,
    )
    obs_arr, _ = env.reset()
    rmse_p_sq = rmse_q_sq = rmse_r_sq = 0.0
    n = 0
    for k in range(n_steps):
        observation = {
            "omega": np.array([obs_arr[2], obs_arr[4], obs_arr[3]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed), "state": obs_arr.copy(),
        }
        refs = {"C_star": 1.0, "phi_cmd": 0.0,
                "beta_cmd": 0.0, "V_cmd": 200.0}
        u_rad = agent.predict(observation, references=refs, time_step=k)
        obs_arr, _r, _term, _trunc, _info = env.step(np.rad2deg(u_rad))
        next_obs = {
            "omega": np.array([obs_arr[2], obs_arr[4], obs_arr[3]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed), "state": obs_arr.copy(),
        }
        agent.learn(next_obs, references=refs, time_step=k)
        # Skip the first 2s of transient before sampling RMSE.
        if k * env.dt >= 2.0:
            rmse_p_sq += float(obs_arr[2] ** 2)
            rmse_q_sq += float(obs_arr[4] ** 2)  # wz = q.
            rmse_r_sq += float(obs_arr[3] ** 2)  # wy = r.
            n += 1
    n = max(n, 1)
    return {
        "p": math.sqrt(rmse_p_sq / n),
        "q": math.sqrt(rmse_q_sq / n),
        "r": math.sqrt(rmse_r_sq / n),
    }


def _emit(rows: list[dict], out_md: Path, out_csv: Path | None) -> None:
    cols = ["method", "scenario", "p_rmse", "q_rmse", "r_rmse"]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# AIDI benchmark report\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write(
                "| " + " | ".join(
                    str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.4f}"
                    for c in cols
                ) + " |\n"
            )
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for r in rows:
                writer.writerow([r[c] for c in cols])


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIDI benchmark CLI")
    parser.add_argument("--env", default="f16_nonlinear_angular")
    parser.add_argument(
        "--baselines", default="frozen",
        help="Comma-separated baseline method ids (currently only 'frozen').",
    )
    parser.add_argument("--scenarios", default="nominal,stab_25")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    if args.env != "f16_nonlinear_angular":
        raise SystemExit(f"unsupported env: {args.env}")

    methods = ["adaptive"] + [
        b.strip() for b in args.baselines.split(",") if b.strip()
    ]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario: {s}")

    alpha_trim, stab_trim = _solve_trim()
    rows: list[dict] = []
    for method in methods:
        for scenario in scenarios:
            agg = {"p": 0.0, "q": 0.0, "r": 0.0}
            for _ in range(args.episodes):
                ep = _run_episode(
                    method, scenario, args.steps, alpha_trim, stab_trim,
                )
                for k in agg:
                    agg[k] += ep[k]
            n = max(args.episodes, 1)
            rows.append({
                "method": method,
                "scenario": scenario,
                "p_rmse": agg["p"] / n,
                "q_rmse": agg["q"] / n,
                "r_rmse": agg["r"] / n,
            })

    _emit(rows, Path(args.out), Path(args.csv) if args.csv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
