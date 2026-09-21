"""Reproduce healthy-only initialization, learning-rate and trigger searches.

The linear stage is a cheap screening model, never final validation. Subsequent
stages run the actual trainable ET-DHP networks on the nonlinear B747.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from example.reinforcement_learning.incremental_adp import (
    example_etdhp_tuned_b747 as tune,
    example_etdhp_vs_pid_b747 as demo,
)


def search_linear(output, count=1600):
    rng = np.random.default_rng(2709)
    a, b = demo.healthy_linearization(0.05)
    n = count
    # The search uses only the nominal healthy plant. Costs are independent of fault traces.
    qbase = np.array([0.1, 0.2, 0.2, 2, 5, 1, 2])
    rbase = np.array([0.2, 0.2])
    lo = np.log10([0.0001, 0.01, 0.01, 1, 1, 0.1, 0.1, 0.005, 0.005])
    hi = np.log10([3, 20, 20, 200, 200, 5000, 5000, 2, 2])
    params = 10 ** rng.uniform(lo, hi, (n, 9))
    params[0] = np.r_[qbase, rbase]
    ks = []
    for v in params:
        q, r = np.diag(v[:7] * 1e-3), np.diag(v[7:] * 1e-3)
        p = solve_discrete_are(a, b, q, r)
        ks.append(np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a))
    k = np.array(ks)
    metrics = []
    for channel, amp in [(4, 1.0), (3, 0.5)]:
        x = np.zeros((n, 7))
        x[:, channel] = -amp
        ref = np.zeros(7)
        ref[channel] = amp
        disturbance = (a - np.eye(7)) @ ref
        disturbance[5:] -= 0.05 * 0.1 * ref[3:5]
        iae = np.zeros(n)
        ise = np.zeros(n)
        energy = np.zeros(n)
        last = np.zeros(n)
        over = np.zeros(n)
        tail = np.zeros(n)
        for j in range(6000):
            u = 8 * np.tanh(-np.einsum("nij,nj->ni", k, x) / 8)
            x = x @ a.T + u @ b.T + disturbance
            error = np.abs(x[:, channel]) / amp
            other = 3 if channel == 4 else 4
            iae += 0.05 * (error + 0.5 * np.abs(x[:, other]) / amp)
            ise += 0.05 * (error**2 + 0.5 * (x[:, other] / amp) ** 2)
            energy += 0.05 * np.sum(u**2, axis=1)
            last = np.where(error > 0.02, (j + 1) * 0.05, last)
            over = np.maximum(over, x[:, channel] / amp)
            if j >= 5000:
                tail += error / 1000
        # Penalize slow/bias cases, overshoot > 8%, and very high command energy.
        score = (
            iae
            + 0.25 * ise
            + 0.15 * last
            + 120 * np.maximum(over - 0.08, 0)
            + 1000 * tail
            + 0.008 * energy
        )
        metrics.append(
            dict(
                score=score,
                iae=iae,
                ise=ise,
                energy=energy,
                settling=last,
                overshoot=over,
                tail=tail,
            )
        )
    score = metrics[0]["score"] + metrics[1]["score"]
    result = []
    for index in np.argsort(score)[:60]:
        result.append(
            {
                "index": int(index),
                "score": float(score[index]),
                "q": params[index, :7].tolist(),
                "r": params[index, 7:].tolist(),
                "cases": [
                    {key: float(val[index]) for key, val in m.items()} for m in metrics
                ],
            }
        )
    report = {
        "seed": 2709,
        "count": count,
        "results": result,
        "original_score": float(score[0]),
        "bounds_log10": {"low": lo.tolist(), "high": hi.tolist()},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "linear-search.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        output / "linear-search-all.npz", parameters=params, score=score
    )
    return report


def _evaluate_job(job):
    name, parameters, response, output = job
    torch.set_num_threads(1)
    settings = tune.Tuning(**parameters)
    results = []
    for channel in (0, 1):
        trace, result = tune.run_case(settings, channel=channel, response=response)
        tune.save_case(Path(output) / f"{name}_{channel}", trace, result)
        results.append(result)
    return dict(
        name=name,
        params=asdict(settings),
        score=sum(tune.tuning_score(r) for r in results),
        results=results,
    )


def run_batch(candidates, output, stage, response, workers=2):
    output = Path(output)
    (output / stage).mkdir(parents=True, exist_ok=True)
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(name, params, response, output / stage) for name, params in candidates]
        for future in as_completed([pool.submit(_evaluate_job, job) for job in jobs]):
            result = future.result()
            outcomes.append(result)
            print(result["name"], result["score"], flush=True)
            outcomes.sort(key=lambda item: item["score"])
            (output / f"{stage}_summary.json").write_text(
                json.dumps(outcomes, indent=2, allow_nan=False) + "\n"
            )
    return outcomes


def shortlist(output, workers=2):
    data = json.loads((Path(output) / "linear-search.json").read_text())
    candidates = [("original_tight_trigger", {})]
    candidates += [
        (f"linear_{v['index']}", dict(q=v["q"], r=v["r"])) for v in data["results"][:8]
    ]
    return run_batch(candidates, output, "search", 200.0, workers)


def refine(output, workers=2):
    data = json.loads((Path(output) / "search_summary.json").read_text())
    base = data[0]["params"]
    candidates = []
    for factor in [1, 3, 10, 30, 100]:
        parameters = copy.deepcopy(base)
        parameters["q"][3] *= factor
        candidates.append((f"roll_weight_{factor}", parameters))
    for rho in [0.0002, 0.005, 0.01]:
        parameters = copy.deepcopy(base)
        parameters["rho"] = rho
        candidates.append((f"rho_{rho}", parameters))
    for lr in [1e-8, 1e-6]:
        parameters = copy.deepcopy(base)
        parameters["actor_lr"], parameters["critic_lr"] = lr, lr * 10
        candidates.append((f"lr_{lr}", parameters))
    for epochs in [3, 10]:
        parameters = copy.deepcopy(base)
        parameters["epochs"] = epochs
        candidates.append((f"epochs_{epochs}", parameters))
    return run_batch(candidates, output, "refine", 300.0, workers)


def balanced_score(result):
    if result["status"] != "complete":
        return 1e6
    m = result["metrics"]
    settling = m["settling_2pct_s"] if m["settling_2pct_s"] is not None else 500
    return (
        sum(result["all_channel_iae"])
        + 300 * (abs(m["tail_bias"]) + m["tail_std"]) / abs(result["amplitude"])
        + 0.05 * settling
        + 0.5 * max(m["overshoot_pct"] - 10, 0)
        + 0.001 * sum(result["command_total_variation"])
    )


def _balanced_job(job):
    name, parameters, output = job
    torch.set_num_threads(1)
    settings = tune.Tuning(**parameters)
    results = []
    for channel in (0, 1):
        trace, result = tune.run_case(
            settings,
            channel=channel,
            response=500.0,
            warmup=300.0,
            step_time=20.0,
            fault_time=137.0,
        )
        tune.save_case(Path(output) / f"{name}_{channel}", trace, result)
        results.append(result)
    return dict(
        name=name,
        params=asdict(settings),
        score=sum(balanced_score(r) for r in results),
        results=results,
    )


def balanced(output, workers=2):
    output = Path(output)
    data = json.loads((output / "search_summary.json").read_text())
    eligible = [
        item for item in data if all(r["status"] == "complete" for r in item["results"])
    ]

    def all_channel_score(item):
        return sum(
            sum(r["all_channel_iae"])
            + 1000
            * (abs(r["metrics"]["tail_bias"]) + r["metrics"]["tail_std"])
            / abs(r["amplitude"])
            for r in item["results"]
        )

    bases = sorted(eligible, key=all_channel_score)[:2]
    folder = output / "balanced"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for source in bases:
        for heading_factor in (1, 3):
            for integral_factor in (0.3, 1, 3):
                params = copy.deepcopy(source["params"])
                params["q"][4] *= heading_factor
                params["q"][5] *= integral_factor
                params["q"][6] *= integral_factor
                params["rho"] = 0.005
                name = f"{source['name']}_h{heading_factor}_i{integral_factor}"
                jobs.append((name, params, folder))
    (folder / "protocol.json").write_text(
        json.dumps(
            {
                "condition": "healthy only",
                "warmup": 300,
                "response": 500,
                "criterion": "balanced_score",
                "candidates": [(j[0], j[1]) for j in jobs],
            },
            indent=2,
        )
        + "\n"
    )
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_balanced_job, job) for job in jobs]):
            result = future.result()
            outcomes.append(result)
            outcomes.sort(key=lambda r: r["score"])
            print(result["name"], result["score"], flush=True)
            (folder / "summary.json").write_text(
                json.dumps(outcomes, indent=2, allow_nan=False) + "\n"
            )
    return outcomes


def _conditioning_job(job):
    name, params, diagnostic_frozen, output = job
    torch.set_num_threads(1)
    settings = tune.Tuning(**params)
    agent, _ = tune.make_agent(settings)
    if diagnostic_frozen:
        # Isolate the cause of failure; excluded from all deployable candidates.
        agent.actor_opt.param_groups[0]["lr"] = 0
        agent.critic_opt.param_groups[0]["lr"] = 0
    cfg = demo.Experiment(
        duration=60.0, fault_time=30.0, initial_heading_deg=2.0, initial_roll_deg=1.0
    )
    trace, diagnostics = demo.rollout(cfg, "etdhp", fault=False, template=agent)
    result = dict(
        **diagnostics,
        parameters=asdict(settings),
        diagnostic_frozen=diagnostic_frozen,
        metrics=demo.metrics(trace, cfg),
    )
    path = Path(output) / name
    np.savetxt(
        path.with_suffix(".csv"),
        trace,
        delimiter=",",
        header=",".join(demo.TRACE_NAMES),
        comments="",
    )
    path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return name, result


def conditioning(output, workers=2):
    output = Path(output)
    selected = json.loads((output / "balanced/summary.json").read_text())[0]["params"]
    folder = output / "conditioning"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [("frozen_diagnostic", selected, True, folder)]
    for scale in (0.001, 0.0001, 0.00001):
        for actor_lr in (1e-7, 3e-8, 1e-8):
            params = copy.deepcopy(selected)
            params.update(cost_scale=scale, actor_lr=actor_lr)
            jobs.append((f"scale_{scale}_actor_{actor_lr}", params, False, folder))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(
            [pool.submit(_conditioning_job, job) for job in jobs]
        ):
            name, result = future.result()
            results[name] = result
            print(name, result["status"], result["steps"], flush=True)
            (folder / "summary.json").write_text(
                json.dumps(results, indent=2, allow_nan=False) + "\n"
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["linear", "shortlist", "refine", "balanced", "conditioning"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/b747-etdhp-vs-pid/tuning")
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "linear":
        search_linear(args.output)
    elif args.stage == "shortlist":
        shortlist(args.output, args.workers)
    elif args.stage == "refine":
        refine(args.output, args.workers)
    elif args.stage == "balanced":
        balanced(args.output, args.workers)
    else:
        conditioning(args.output, args.workers)


if __name__ == "__main__":
    main()
