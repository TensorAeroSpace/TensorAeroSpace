"""LQR–MRAC with online and frozen adaptation on the nonlinear F-16.

Run from the repository root:
    python -m example.lqr_controllers.example_lqr_mrac_f16 --output outputs/lqr-mrac-f16
"""

from pathlib import Path
import argparse

from example.lqr_controllers.example_lqr_f16 import (
    Experiment,
    make_setup,
    rollout,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/lqr-mrac-f16"))
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    setup = make_setup()
    traces, reports = {}, {}
    for method in ("lqr", "mrac_frozen", "mrac_online"):
        for fault in (False, True):
            cfg = Experiment(
                duration=args.duration,
                fault_time=args.duration / 2,
                loss_fraction=0.3 if fault else 0.0,
            )
            name = f"{method}_{'fault' if fault else 'nominal'}"
            trace, report = rollout(setup, cfg, method)
            traces[name], reports[name] = trace, report
    save_results(args.output, traces, reports)


if __name__ == "__main__":
    main()
