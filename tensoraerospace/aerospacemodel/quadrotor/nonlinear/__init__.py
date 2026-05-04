"""Pure-numpy 6-DoF nonlinear quadrotor model.

State (12):
    [x_e, y_e, z_e, u_b, v_b, w_b, phi, theta, psi, p_b, q_b, r_b]

  - position in NED earth frame (m)
  - velocity in body frame (m/s)
  - Euler angles ZYX 321 (rad)
  - angular rates in body frame (rad/s)

Control (4): [T, tau_x, tau_y, tau_z]

  - T  : collective thrust along body -z axis (N, positive = up when level)
  - tau_x, tau_y, tau_z : body-frame torques about roll, pitch, yaw axes (N·m)

The control vector is "abstract" — collective thrust and body torques as
seen by the inner control loop, after the motor-mixing / allocation step.
A separate top-level helper maps 4 motor RPMs (or normalised PWM) into
this (T, tau) vector when needed.
"""

from .initial import initial_state, initial_state_dict, set_initial_state
from .model import NonlinearQuadrotor

__all__ = [
    "NonlinearQuadrotor",
    "initial_state",
    "initial_state_dict",
    "set_initial_state",
]
