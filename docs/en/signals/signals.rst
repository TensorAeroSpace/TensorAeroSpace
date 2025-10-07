Signals
=======

TensorAeroSpace provides **17 types of signals** for comprehensive system testing and analysis.

Basic Signals
-------------

Step Signal
~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.unit_step

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import unit_step

    dt = 0.01
    tp = generate_time_period(tn=20)
    tp_unit = unit_step(degree=5, tp=tp, time_step=10, output_rad=False)

.. image:: img/unit_step.png
        :alt: Generated step signal


Ramp Signal
~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.ramp

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import ramp

    tp = generate_time_period(tn=20)
    signal = ramp(tp, slope=0.5, time_start=2.0)

.. image:: img/ramp.png
        :alt: Ramp signal


Pulse Signal
~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.pulse

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import pulse

    tp = generate_time_period(tn=20)
    signal = pulse(tp, amplitude=5.0, time_start=5.0, width=3.0)

.. image:: img/pulse.png
        :alt: Pulse signal


Constant Signal
~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.constant_line

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import constant_line

    tp = generate_time_period(tn=20)
    signal = constant_line(tp, value_state=3.0)

.. image:: img/constant_line.png
        :alt: Constant signal


Periodic Signals
----------------

Sinusoidal Signal
~~~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.sinusoid

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import sinusoid

    dt = 0.01
    tp = generate_time_period(tn=20)
    tp_sinusoid = sinusoid(tp=tp, amplitude=10, frequency=0.01)

.. image:: img/sinusoid.png
        :alt: Sinusoidal signal


Sinusoid with Vertical Shift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.sinusoid_vertical_shift

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import sinusoid_vertical_shift

    tp = generate_time_period(tn=20)
    signal = sinusoid_vertical_shift(tp, frequency=0.5, amplitude=2.0, vertical_shift=5.0)

.. image:: img/sinusoid_vertical_shift.png
        :alt: Sinusoid with vertical shift


Square Wave
~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.square_wave

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import square_wave

    tp = generate_time_period(tn=20)
    signal = square_wave(tp, frequency=0.5, amplitude=3.0, duty_cycle=0.5)

.. image:: img/square_wave.png
        :alt: Square wave signal


Triangular Wave
~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.triangular_wave

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import triangular_wave

    tp = generate_time_period(tn=20)
    signal = triangular_wave(tp, frequency=0.3, amplitude=4.0)

.. image:: img/triangular_wave.png
        :alt: Triangular wave signal


Sawtooth Wave
~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.sawtooth

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import sawtooth

    tp = generate_time_period(tn=20)
    signal = sawtooth(tp, frequency=0.4, amplitude=3.0)

.. image:: img/sawtooth.png
        :alt: Sawtooth signal


Complex Signals
---------------

Chirp Signal
~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.chirp

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import chirp

    tp = generate_time_period(tn=20)
    signal = chirp(tp, f0=0.1, f1=2.0, amplitude=2.0, method='linear')

.. image:: img/chirp.png
        :alt: Chirp signal


Doublet
~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.doublet

**Example**

.. code-block:: python

    import numpy as np
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import doublet

    tp = generate_time_period(tn=20)
    signal = doublet(tp, amplitude=np.deg2rad(10), time_start=5.0, width=1.0)

.. image:: img/doublet.png
        :alt: Doublet signal


Multi-Step Signal
~~~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.multi_step

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import multi_step

    tp = generate_time_period(tn=20)
    signal = multi_step(tp, step_times=[2, 5, 8, 12, 16], step_values=[1, 2, -1, 3, -2])

.. image:: img/multi_step.png
        :alt: Multi-step signal


Exponential Signal
~~~~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.exponential

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import exponential

    tp = generate_time_period(tn=20)
    signal = exponential(tp, amplitude=10.0, time_constant=2.0, time_start=3.0)

.. image:: img/exponential.png
        :alt: Exponential signal


Gaussian Pulse
~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.gaussian_pulse

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import gaussian_pulse

    tp = generate_time_period(tn=20)
    signal = gaussian_pulse(tp, amplitude=8.0, center=10.0, width=1.5)

.. image:: img/gaussian_pulse.png
        :alt: Gaussian pulse signal


Multisine
~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.multisine

**Example**

.. code-block:: python

    import numpy as np
    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import multisine

    tp = generate_time_period(tn=20)
    signal = multisine(tp, frequencies=[0.2, 0.5, 1.0, 1.5], 
                       amplitudes=[2.0, 1.5, 1.0, 0.5],
                       phases=[0, np.pi/4, np.pi/2, np.pi])

.. image:: img/multisine.png
        :alt: Multisine signal


Damped Sinusoid
~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.standart.damped_sinusoid

**Example**

.. code-block:: python

    from tensoraerospace.utils import generate_time_period
    from tensoraerospace.signals.standart import damped_sinusoid

    tp = generate_time_period(tn=20)
    signal = damped_sinusoid(tp, frequency=1.0, amplitude=5.0, damping=0.3, time_start=2.0)

.. image:: img/damped_sinusoid.png
        :alt: Damped sinusoid signal


Random Signals
--------------

Random Signal by Frequency and Amplitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tensoraerospace.signals.random.full_random_signal

**Example**

.. code-block:: python

    from tensoraerospace.signals.random import full_random_signal

    signal = full_random_signal(0, 0.01, 20, (-0.5, 0.5), (-0.5, 0.5))

.. image:: img/full_random.png
        :alt: Random signal by frequency and amplitude
