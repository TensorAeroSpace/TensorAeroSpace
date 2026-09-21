#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of using the improved benchmark for control-system analysis.

This file shows how to use the new features of the ControlBenchmark class
to build nice, informative plots for control-quality analysis.
"""

import numpy as np

from tensoraerospace.benchmark import ControlBenchmark


def generate_sample_system_response(
    time, overshoot=0.2, settling_time=3.0, noise_level=0.01
):
    """
    Generate a sample control-system response.

    Args:
        time (np.ndarray): Time array
        overshoot (float): Overshoot (0.0 - 1.0)
        settling_time (float): Approximate 2% settling time after the step
        noise_level (float): Noise level

    Returns:
        tuple: (control_signal, system_signal)
    """
    if not 0 <= overshoot < 1:
        raise ValueError("overshoot must be in [0, 1)")
    if settling_time <= 0:
        raise ValueError("settling_time must be positive")
    time = np.asarray(time, dtype=float)
    control_signal = (time >= 1.0).astype(float)
    elapsed = np.maximum(time - 1.0, 0.0)

    if overshoot == 0:
        # Critical damping: (1 + wn*t)*exp(-wn*t) is below 2% at wn*t=6.
        wn = 6.0 / settling_time
        system_signal = 1 - (1 + wn * elapsed) * np.exp(-wn * elapsed)
    else:
        zeta = -np.log(overshoot) / np.sqrt(np.pi**2 + np.log(overshoot) ** 2)
        # Ts ~= 4/(zeta*wn), rather than 4/wn.
        wn = 4.0 / (zeta * settling_time)
        wd = wn * np.sqrt(1 - zeta**2)
        system_signal = 1 - np.exp(-zeta * wn * elapsed) * (
            np.cos(wd * elapsed) + (zeta * wn / wd) * np.sin(wd * elapsed)
        )

    # Add noise
    system_signal += np.random.normal(0, noise_level, len(system_signal))

    return control_signal, system_signal


def main():
    """
    Main function demonstrating the benchmark capabilities.
    """
    print("🚀 TensorAeroSpace improved benchmark demo")
    print("=" * 60)

    # Build the time array
    dt = 0.01
    time = np.arange(0, 10, dt)

    # Create the benchmark instance
    benchmark = ControlBenchmark()

    print("\n1️⃣  Analysis of a single control system")
    print("-" * 40)

    # Generate data for a single system
    control_signal, system_signal = generate_sample_system_response(
        time, overshoot=0.15, settling_time=2.5, noise_level=0.005
    )

    # Build a nice plot
    metrics = benchmark.plot(
        control_signal,
        system_signal,
        signal_val=0.5,
        dt=dt,
        tps=time,
        title="PID controller analysis",
    )

    # Generate the report
    report = benchmark.generate_report(
        control_signal,
        system_signal,
        signal_val=0.5,
        dt=dt,
        system_name="PID controller",
    )
    print(report)

    print("\n2️⃣  Comparison of several control systems")
    print("-" * 50)

    # Build data for comparing several systems
    systems_data = {}

    # System 1: Fast with overshoot
    control1, system1 = generate_sample_system_response(
        time, overshoot=0.25, settling_time=1.5, noise_level=0.003
    )
    systems_data["Fast system"] = {
        "control_signal": control1,
        "system_signal": system1,
        "time": time,
    }

    # System 2: Slow without overshoot
    control2, system2 = generate_sample_system_response(
        time, overshoot=0.05, settling_time=4.0, noise_level=0.002
    )
    systems_data["Slow system"] = {
        "control_signal": control2,
        "system_signal": system2,
        "time": time,
    }

    # System 3: Optimal
    control3, system3 = generate_sample_system_response(
        time, overshoot=0.10, settling_time=2.0, noise_level=0.004
    )
    systems_data["Optimal system"] = {
        "control_signal": control3,
        "system_signal": system3,
        "time": time,
    }

    # Compare the systems
    all_metrics = benchmark.compare_systems(systems_data, signal_val=0.5, dt=dt)

    # Print the comparison table to the console
    print("\n📊 Metric comparison table:")
    print("-" * 80)
    print(
        f"{'System':<20} {'Overshoot%':<12} {'Settling t.':<12} {'Damping':<12} {'Steady err.':<12}"
    )
    print("-" * 80)

    for system_name, metrics in all_metrics.items():
        settling = (
            f"{metrics['settling_time']:.3f}"
            if metrics["settling_time"] is not None
            else "N/A"
        )
        print(
            f"{system_name:<20} {metrics['overshoot']:<12.2f} "
            f"{settling:<12} {metrics['damping_degree']:<12.3f} "
            f"{metrics['static_error']:<12.4f}"
        )

    print("\n✅ Demo finished!")
    print("\n💡 Improved benchmark features:")
    print("   • Nice plots with a modern design")
    print("   • Automatic annotations of key points")
    print("   • Quality-metric tables")
    print("   • Multi-system comparison")
    print("   • Control-error plots")
    print("   • Text reports with ratings")
    print("   • Customizable color schemes")


if __name__ == "__main__":
    main()
