"""Run the public SDK's B747 engine-loss comparison.

Install tensoraerospace, then run this file from any working directory.
See the companion notebook for plots and a walkthrough of the protocol.
"""

from tensoraerospace.benchmark import B747EngineFailureBenchmark


def main():
    benchmark = B747EngineFailureBenchmark()
    settings, trials = benchmark.tune_baselines()
    print(f"Evaluated {len(trials)} candidates on the healthy aircraft")
    for algorithm in benchmark.algorithms:
        result = benchmark.run(algorithm, fault=True, **settings.get(algorithm, {}))
        print(algorithm, result["after"])


if __name__ == "__main__":
    main()
