import os

import matlab.engine
import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase


class AngularF16(ModelBase):
    """High-maneuverability F-16 aircraft control object in angular coordinates.

    Action space:
        * stab_act: Elevator [rad]
        * ail_act: Ailerons [rad]
        * dir_act: Rudder [rad]

    State space:
        * alpha: Angle of attack [rad]
        * beta: Sideslip angle [rad]
        * wx: Roll angular velocity [rad/s]
        * wy: Yaw angular velocity [rad/s]
        * wz: Pitch angular velocity [rad/s]
        * gamma: Roll angle [rad]
        * psi: Yaw angle [rad]
        * theta: Pitch angle [rad]
        * stab: Elevator position [rad]
        * ail: Aileron position [rad]
        * dir: Rudder position [rad]
        * dstab: Elevator angular velocity [rad/s]
        * dail: Aileron angular velocity [rad/s]
        * ddir: Rudder angular velocity [rad/s]

    Usage example:

    >>> from aerospacemodel.model.f16.nonlinear.angular import initial_state
    >>> model = AngularF16(initial_state)
    >>> x_t = model.run_step([ [0], [0], [0] ])

    Args:
        x0: Initial state.
        t0: (Optional) Initial time.
        dt: (Optional) Discretization step.

    """

    def __init__(self, x0, selected_state_output=None, t0=0, dt: float = 0.01):
        super(AngularF16, self).__init__(x0, selected_state_output, t0, dt)
        self.matlab_files_path = os.path.join(os.path.dirname(__file__), "matlab_code")
        self.eng = matlab.engine.start_matlab()  # Запуск экземпляр Matlab
        self.eng.addpath(self.matlab_files_path)
        self.list_state = [
            "alpha",
            "beta",
            "wx",
            "wy",
            "wz",
            "gamma",
            "psi",
            "theta",
            "stab",
            "dstab",
            "ail",
            "dail",
            "dir",
            "ddir",
        ]
        self.control_list = ["stab", "ail", "dir"]
        self.action_space_length = len(self.control_list)
        self.param = (
            self.eng.airplane_parameters()
        )  # Получаем параметры объекта управления
        self.x_history = [x0]
        self._initialize_selected_state_index(
            self.selected_state_output, self.list_state
        )

    def get_param(self):
        """Get control object parameters.

        Returns:
            Control object parameters.
        """
        return self.param

    def set_param(self, new_param):
        """Set new control object parameters.

        Args:
           new_param: Control object parameters.
        """
        self.param = new_param

    def run_step(self, u: matlab.double):
        """Calculate control object state.

        Control signal format:

        >>> stab_act, ail_act, dir_act = 0,0,0
        >>> [
        >>>    [stab_act],
        >>>    [ail_act],
        >>>    [dir_act]
        >>> ]

        Args:
            u: Control signal.

        Returns:
            Control object state.

        Usage example:
        >>> from aerospacemodel.model.f16.nonlinear.angular import initial_state
        >>> model = AngularF16(initial_state)
        >>> xt = model.run_step([ [0], [0], [0] ])
        """
        if not isinstance(u, matlab.double):
            u = matlab.double(u)
        if len(list(u)) != self.action_space_length:
            raise Exception(
                "Размерность управляющего вектора задана неверно."
                + f" Текущее значение {len(list(u))}, не соответсвует {self.action_space_length}"
            )
        x_t = self.eng.step(
            self.x_history[-1], self.dt, u, self.t0, self.time_step, self.param
        )
        self.x_history.append(x_t)
        self.u_history.append(u)
        self.time_step += 1
        if self.selected_state_output:
            return np.array(x_t[self.selected_state_index])
        return np.array(x_t)
