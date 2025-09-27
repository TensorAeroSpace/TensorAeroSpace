"""Signal generation module for testing and training.

This module provides various types of signals for use in simulations
and control system testing, including random signals, sinusoidal
signals and unit step functions.
"""

from .random import full_random_signal as full_random_signal
from .standart import sinusoid as sinusoid
from .standart import unit_step as unit_step
