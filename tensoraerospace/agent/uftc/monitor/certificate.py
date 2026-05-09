"""Numerical certificate of Lemma 4.1 hypotheses + empirical pass-rate.

Standalone callable suitable for an offline CLI:

    python -m tensoraerospace.agent.uftc.monitor.certificate \
        --config artifacts/uftc/cfg.yaml \
        --rollouts artifacts/uftc/cert_rollouts.npz \
        --report artifacts/uftc/uub_certificate.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CertificateReport:
    metzler_check: str
    hurwitz_check: str
    lambda_min: float
    mu_uub_pred: float
    rollouts: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdict: str = "pending"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_certificate(
    cfg: dict,
    *,
    rollouts: dict[str, np.ndarray] | None = None,
    transient_steps: int = 200,
    pass_rate_target: float = 0.99,
) -> CertificateReport:
    eps = np.asarray(cfg["eps_matrix"], dtype=np.float64)
    a = np.asarray(cfg["a_diag"], dtype=np.float64)
    d = np.asarray(cfg["d_disturbance"], dtype=np.float64)
    c = np.asarray(cfg["c_weights"], dtype=np.float64)

    metzler = "pass" if (eps - np.diag(np.diag(eps)) >= -1e-12).all() else "fail"
    M = np.diag(a) - eps
    eigvals = np.linalg.eigvals(M)
    lambda_min = float(np.min(np.real(eigvals)))
    hurwitz = "pass" if lambda_min > 0 else "fail"

    if metzler == "pass" and hurwitz == "pass":
        sol = np.linalg.solve(M, d)
        mu = float(np.dot(c, np.abs(sol)))
    else:
        mu = float("nan")

    rollouts = rollouts or {}
    rollouts_out: dict[str, dict[str, Any]] = {}
    if rollouts and not np.isnan(mu):
        for name, arr in rollouts.items():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim != 2:
                continue
            tail = arr[:, transient_steps:] if arr.shape[1] > transient_steps else arr
            n = arr.shape[0]
            ok = (tail.max(axis=1) <= mu).sum()
            rollouts_out[name] = {
                "n": int(n),
                "transient_steps": int(transient_steps),
                "pass_rate": float(ok / max(n, 1)),
                "max_v_total": float(arr.max()),
            }
        worst = min((r["pass_rate"] for r in rollouts_out.values()), default=1.0)
        verdict = (
            "pass"
            if (metzler == "pass" and hurwitz == "pass" and worst >= pass_rate_target)
            else "fail"
        )
    else:
        verdict = "pass" if (metzler == "pass" and hurwitz == "pass") else "fail"

    return CertificateReport(
        metzler_check=metzler,
        hurwitz_check=hurwitz,
        lambda_min=lambda_min,
        mu_uub_pred=mu,
        rollouts=rollouts_out,
        verdict=verdict,
    )


def _cli() -> None:  # pragma: no cover - CLI plumbing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rollouts", type=str, default=None)
    parser.add_argument("--report", type=str, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rollouts = {}
    if args.rollouts:
        npz = np.load(args.rollouts)
        rollouts = {k: npz[k] for k in npz.files}
    rep = run_certificate(cfg, rollouts=rollouts)
    Path(args.report).write_text(rep.to_json())
    raise SystemExit(0 if rep.verdict == "pass" else 1)


if __name__ == "__main__":  # pragma: no cover
    _cli()
