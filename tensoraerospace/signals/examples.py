"""Examples of using control signals for testing and training.

This module demonstrates how to use various control signals
for system identification, control system testing, and reinforcement learning.
"""

import numpy as np
from . import (
    chirp,
    constant_line,
    damped_sinusoid,
    doublet,
    exponential,
    full_random_signal,
    gaussian_pulse,
    multi_step,
    multisine,
    pulse,
    ramp,
    sawtooth,
    sinusoid,
    sinusoid_vertical_shift,
    square_wave,
    triangular_wave,
    unit_step,
)


def example_basic_signals():
    """Example of basic control signals."""
    # Time array
    t = np.linspace(0, 10, 1000)
    
    # Step response test
    step_signal = unit_step(t, degree=5, time_step=2)
    
    # Ramp input
    ramp_signal = ramp(t, slope=0.5, time_start=1.0)
    
    # Pulse for transient analysis
    pulse_signal = pulse(t, amplitude=10, time_start=2, width=2)
    
    # Constant reference
    const_signal = constant_line(t, value_state=3.5)
    
    return t, step_signal, ramp_signal, pulse_signal, const_signal


def example_periodic_signals():
    """Example of periodic control signals."""
    t = np.linspace(0, 20, 2000)
    
    # Sinusoidal tracking
    sine_signal = sinusoid(t, frequency=2, amplitude=0.5)
    
    # Sinusoid with DC offset
    sine_offset = sinusoid_vertical_shift(t, frequency=0.5, amplitude=2, vertical_shift=5)
    
    # Square wave for switching control
    square_signal = square_wave(t, frequency=0.5, amplitude=5)
    
    # Triangular wave
    triangle_signal = triangular_wave(t, frequency=0.3, amplitude=3)
    
    # Sawtooth for linear sweep
    sawtooth_signal = sawtooth(t, frequency=0.2, amplitude=4)
    
    return t, sine_signal, sine_offset, square_signal, triangle_signal, sawtooth_signal


def example_frequency_analysis():
    """Example of signals for frequency response analysis."""
    t = np.linspace(0, 50, 5000)
    
    # Linear chirp for frequency sweep
    chirp_linear = chirp(t, f0=0.1, f1=2.0, amplitude=5, method="linear")
    
    # Exponential chirp
    chirp_exp = chirp(t, f0=0.1, f1=2.0, amplitude=5, method="exponential")
    
    # Multi-sine for multi-frequency excitation
    multisine_signal = multisine(
        t,
        frequencies=[0.1, 0.3, 0.8, 1.5],
        amplitudes=[2, 1.5, 1, 0.5],
        phases=[0, np.pi/4, np.pi/2, np.pi]
    )
    
    return t, chirp_linear, chirp_exp, multisine_signal


def example_transient_analysis():
    """Example of signals for transient response analysis."""
    t = np.linspace(0, 20, 2000)
    
    # Doublet for stability analysis (common in aerospace)
    doublet_signal = doublet(t, amplitude=10, time_start=5, width=1)
    
    # Multi-step for step tracking performance
    multistep_signal = multi_step(
        t,
        step_times=[2, 5, 8, 12, 15],
        step_values=[1, 2, -1, 3, -0.5]
    )
    
    # Exponential reference for smooth tracking
    exp_signal = exponential(t, amplitude=10, time_constant=2, time_start=1)
    
    # Gaussian pulse for impulse response
    gauss_signal = gaussian_pulse(t, amplitude=15, center=10, width=1.5)
    
    # Damped sinusoid for oscillatory response
    damped_sine = damped_sinusoid(t, frequency=1, amplitude=8, damping=0.2, time_start=2)
    
    return t, doublet_signal, multistep_signal, exp_signal, gauss_signal, damped_sine


def example_reinforcement_learning_signals():
    """Example of signals for RL training and testing."""
    # Random signal for exploration
    random_signal = full_random_signal(
        t0=0, dt=0.01, tn=100,
        sd=(1, 5),  # Duration between 1 and 5 seconds
        sv=(-10, 10)  # Values between -10 and 10
    )
    
    # Combined signal for diverse training
    t = np.linspace(0, 100, 10000)
    
    # Mix different signals for robust training
    training_signal = (
        unit_step(t, degree=5, time_step=10) +
        sinusoid_vertical_shift(t, frequency=0.1, amplitude=2, vertical_shift=0) +
        pulse(t, amplitude=3, time_start=50, width=5)
    )
    
    return random_signal, t, training_signal


def example_aerospace_maneuvers():
    """Example of typical aerospace maneuver signals."""
    t = np.linspace(0, 30, 3000)
    
    # Elevator doublet for pitch response testing
    elevator_doublet = doublet(t, amplitude=np.deg2rad(5), time_start=5, width=2)
    
    # Aileron pulse for roll maneuver
    aileron_pulse = pulse(t, amplitude=np.deg2rad(10), time_start=10, width=3)
    
    # Rudder sinusoid for Dutch roll excitation
    rudder_sine = sinusoid_vertical_shift(t, frequency=0.3, amplitude=np.deg2rad(3), vertical_shift=0)
    
    # Throttle ramp for acceleration test
    throttle_ramp = ramp(t, slope=0.05, time_start=2)
    throttle_ramp = np.clip(throttle_ramp, 0, 1)  # Limit to [0, 1]
    
    # Multi-step altitude command
    altitude_command = multi_step(
        t,
        step_times=[5, 10, 15, 20, 25],
        step_values=[100, 150, -50, 200, -100]  # meters
    )
    
    return t, elevator_doublet, aileron_pulse, rudder_sine, throttle_ramp, altitude_command


def example_system_identification():
    """Example of signals for system identification."""
    t = np.linspace(0, 100, 10000)
    
    # PRBS-like using random signal
    prbs_like = full_random_signal(
        t0=0, dt=0.01, tn=100,
        sd=(0.5, 2),  # Short duration steps
        sv=(-1, 1)  # Binary-like values
    )
    
    # Multi-sine for frequency domain identification
    multisine_id = multisine(
        t,
        frequencies=np.logspace(-2, 1, 10),  # Logarithmic spacing
        amplitudes=[1.0] * 10,
        phases=np.random.uniform(0, 2*np.pi, 10)  # Random phases
    )
    
    # Chirp for frequency response function
    chirp_id = chirp(t, f0=0.01, f1=5.0, amplitude=3, method="exponential")
    
    return t, prbs_like, multisine_id, chirp_id


if __name__ == "__main__":
    """Run examples and display basic information."""
    print("Control Signals Examples")
    print("=" * 50)
    
    print("\n1. Basic Signals")
    t, step, ramp_sig, pulse_sig, const = example_basic_signals()
    print(f"   Generated {len(t)} time points from {t[0]:.2f} to {t[-1]:.2f}s")
    print(f"   Step signal range: [{step.min():.2f}, {step.max():.2f}]")
    print(f"   Ramp signal range: [{ramp_sig.min():.2f}, {ramp_sig.max():.2f}]")
    
    print("\n2. Periodic Signals")
    t, sine, sine_off, square, triangle, saw = example_periodic_signals()
    print(f"   Generated {len(t)} time points")
    print(f"   Sinusoid amplitude: ±{np.abs(sine).max():.2f}")
    
    print("\n3. Frequency Analysis Signals")
    t, chirp_lin, chirp_e, multisine_sig = example_frequency_analysis()
    print(f"   Chirp signals for frequency sweep")
    print(f"   Multisine with multiple frequency components")
    
    print("\n4. Transient Analysis Signals")
    t, doublet_sig, multistep, exp, gauss, damped = example_transient_analysis()
    print(f"   Doublet for stability testing")
    print(f"   Multi-step for tracking performance")
    
    print("\n5. Aerospace Maneuver Signals")
    t, elev, ail, rud, throt, alt = example_aerospace_maneuvers()
    print(f"   Elevator doublet: ±{np.rad2deg(np.abs(elev).max()):.2f}°")
    print(f"   Aileron pulse: {np.rad2deg(ail.max()):.2f}°")
    
    print("\n6. System Identification Signals")
    t, prbs, ms_id, chirp_id_sig = example_system_identification()
    print(f"   PRBS-like random signal")
    print(f"   Multi-sine for frequency domain ID")
    
    print("\n" + "=" * 50)
    print("All examples generated successfully!")
    print("\nAvailable signals:")
    signals_list = [
        "unit_step", "ramp", "pulse", "constant_line",
        "sinusoid", "sinusoid_vertical_shift", "square_wave",
        "triangular_wave", "sawtooth", "chirp", "multisine",
        "doublet", "multi_step", "exponential", "gaussian_pulse",
        "damped_sinusoid", "full_random_signal"
    ]
    for i, sig in enumerate(signals_list, 1):
        print(f"  {i:2d}. {sig}")
