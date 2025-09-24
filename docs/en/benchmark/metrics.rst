Metrics
=======

Steady-State Error
~~~~~~~~~~~~~~~~~~

Steady-state error is the difference between the steady-state value of the system output and the desired value. It is usually evaluated once the system reaches steady state after an input change or disturbance.

.. math::
    e_{ss} = u(t) - y(t)

where :math:`u(t)` is the reference (control input) and :math:`y(t)` is the system output.

Steady-state error reflects the ability of a control system to match the desired output. A zero steady-state error means the system perfectly tracks the target without residual deviation. A nonzero error indicates shortcomings in the control loop, such as modeling inaccuracies, calibration issues, nonlinearities, or external disturbances. Thus, steady-state error is an important performance indicator and helps identify improvement opportunities.

.. autofunction:: tensoraerospace.benchmark.function.static_error


Damping Degree of the Transient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimating the damping degree of a transient response can be challenging, especially for oscillatory or aperiodic systems. A straightforward approach is to compute the relative decrease in successive oscillation peaks:

.. math::
    y = 1 - \frac{A_n}{A_{n-1}}

where :math:`y` is the damping degree, :math:`A_n` is the amplitude of the nth peak, and :math:`A_{n-1}` is the amplitude of the previous peak. Larger values of :math:`y` correspond to stronger damping.

This method assumes that the peaks can be identified. If the peaks are ambiguous or difficult to detect, the method may not work effectively. The damping degree characterizes how quickly the control system eliminates oscillations and reaches steady state. Higher damping generally implies faster suppression of oscillations, yet excessive damping may slow the response. Therefore, the optimal damping level depends on the system requirements.

.. autofunction:: tensoraerospace.benchmark.function.damping_degree


Settling Time
~~~~~~~~~~~~~

Settling time is the time required for a control system to reach and stay within a specified tolerance band (commonly ±5% or ±2%) around the steady-state value after an input change or disturbance.

A practical algorithm:

- Determine the steady-state value :math:`Y_{final}`, for example by averaging the final portion of the response.
- Find the first instant when the response enters the tolerance band.
- Find the last time the response leaves the band.
- Settling time is the difference between these time instants.

Settling time measures how quickly the system adapts to new conditions. Shorter settling times indicate faster responses, but excessively short times may lead to instability or oscillations. Tuning the settling time helps balance rapid response with stability.

.. autofunction:: tensoraerospace.benchmark.function.settling_time


Overshoot
~~~~~~~~~

Overshoot is the amount by which the peak value of the system response exceeds the steady-state value, typically expressed as a percentage:

.. math::
    O = \frac{M - Y_{final}}{Y_{final}} \times 100\%

where :math:`O` is the overshoot percentage, :math:`M` is the peak response value, and :math:`Y_{final}` is the steady-state value. For oscillatory systems, :math:`Y_{final}` may be approximated by the average after the response settles.

Overshoot indicates how smoothly the system reaches the steady state. Lower overshoot means fewer oscillations. Excessive overshoot may signal instability or insufficient damping. The ideal level depends on application-specific requirements.

.. autofunction:: tensoraerospace.benchmark.function.overshoot


