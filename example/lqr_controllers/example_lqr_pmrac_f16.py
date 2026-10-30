"""LQR, conventional MRAC and predictor-based MRAC on the nonlinear F-16.

Run from the repository root:
    python -m example.lqr_controllers.example_lqr_pmrac_f16 --output outputs/lqr-pmrac-f16
"""

import argparse
from pathlib import Path

from example.lqr_controllers.example_lqr_f16 import (
    Experiment,
    make_setup,
    rollout,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/lqr-pmrac-f16"))
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    setup = make_setup()
    traces, reports = {}, {}
    methods = ("lqr", "mrac_frozen", "mrac_online", "pmrac_frozen", "pmrac_online")
    for method in methods:
        for fault in (False, True):
            cfg = Experiment(
                duration=args.duration,
                fault_time=args.duration / 2,
                loss_fraction=0.3 if fault else 0.0,
                fault_kind="actuator" if fault else "section_loss",
            )
            name = f"{method}_{'fault' if fault else 'nominal'}"
            trace, report = rollout(setup, cfg, method)
            traces[name], reports[name] = trace, report
    save_results(args.output, traces, reports)


if __name__ == "__main__":
    main()
