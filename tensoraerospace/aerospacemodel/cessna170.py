import numpy as np

from tensoraerospace.aerospacemodel.base import ModelBase


class LongitudinalCessna170(ModelBase):
    """Cessna 170 nonlinear model in longitudinal control channel.

    Notes:
        - States: `u` (forward velocity, m/s), `w` (vertical velocity, m/s),
          `q` (pitch rate, rad/s), `theta` (pitch angle, rad)
        - Inputs: `ele` (elevator, rad), `throttle` (0..1)
        - Aerodynamic model: classic CL/CD/Cm with q-hat damping and
          elevator effectiveness. Coefficients are approximate placeholders
          suitable for simulation and can be replaced with a Digital
          DATCOM/flight-test bank.
    """

    def __init__(
        self,
        x0: "np.ndarray | list[float]",
        number_time_steps: int,
        selected_state_output: "list[int] | None" = None,
        t0: float = 0,
        dt: float = 0.01,
        aero_params: "dict | None" = None,
    ) -> None:
        super().__init__(x0, selected_state_output, t0, dt)

        # States and controls
        self.selected_states = ["u", "w", "q", "theta"]
        self.list_state = self.selected_states
        self.control_list = ["ele", "throttle"]

        # NOTE: ModelBase._initialize_selected_state_index initializes a few
        # bookkeeping arrays but also resets `list_state`/`control_list`.
        # This model's run_step relies on `control_list`, so we restore them
        # after the call.
        self._initialize_selected_state_index(
            self.selected_state_output, self.list_state
        )
        self.list_state = self.selected_states
        self.control_list = ["ele", "throttle"]

        # Histories
        self.x_history = [np.array(x0, dtype=float).reshape(-1)]
        self.u_history = []

        # Physical and aero parameters (can be overridden via aero_params)
        self.params = {
            # Atmosphere and geometry
            "rho": 1.225,  # kg/m^3
            "S": 16.2,  # m^2 (approx wing area)
            "cbar": 1.5,  # m (approx mean aerodynamic chord)
            # Inertial and mass
            "m": 1000.0,  # kg (approx mass)
            "Iyy": 1100.0,  # kg*m^2 (approx pitch inertia)
            # Aerodynamic coefficients (dimensionless)
            "CL0": 0.3,
            "CL_alpha": 5.2,  # per rad
            "CL_de": 0.30,  # per rad
            "CL_q": 7.5,  # per (q-hat)
            "CD0": 0.030,
            "k_ind": 0.055,  # induced drag factor ~ 1/(pi e AR)
            "Cm0": -0.02,
            "Cm_alpha": -0.5,  # per rad
            "Cm_de": -1.0,  # per rad
            "Cm_q": -8.0,  # per (q-hat)
            # Thrust model
            "T_static": 4500.0,  # N at throttle=1, V->0 (approx)
            "V_max": 75.0,  # m/s speed where thrust ~ 0 in simple model
            # Gravity
            "g": 9.81,
            # Input limits (magnitude and rate)
            "ele_lim": np.deg2rad(25.0),  # rad
            "ele_rate": np.deg2rad(60.0),  # rad/s
            "thr_min": 0.0,
            "thr_max": 1.0,
            "thr_rate": 0.5,  # 1/s
        }
        if aero_params is not None:
            self.params.update(aero_params)

        # Simulation bookkeeping
        self.number_time_steps = number_time_steps
        self.time_step = 0

    # ---------- Public API ----------
    def get_param(self) -> dict:
        """Return current model parameters."""
        return self.params

    def set_param(self, new_param: dict) -> None:
        """Replace model parameters with a new dictionary."""
        self.params = new_param

    def run_step(self, u: "np.ndarray | list[float]"):
        """Advance the simulation by one time step.

        Args:
            u_t: [ele, throttle]

        Returns:
            Next state vector (optionally filtered by selected_state_output).
        """
        u_t = np.array(u, dtype=float).reshape(-1)
        if u_t.size != len(self.control_list):
            raise ValueError(
                (
                    "Размерность управляющего вектора задана неверно. "
                    f"Текущее значение {u_t.size}, не соответствует "
                    f"{len(self.control_list)}"
                )
            )

        # Previous control for rate limiting
        if self.u_history:
            u_prev = np.array(self.u_history[-1], dtype=float).reshape(-1)
        else:
            u_prev = u_t.copy()

        # Apply input magnitude and rate limits
        ele_cmd = self._limit_with_rate(
            value=u_t[0],
            prev=u_prev[0],
            mag_min=-self.params["ele_lim"],
            mag_max=self.params["ele_lim"],
            rate=self.params["ele_rate"],
        )
        thr_cmd = self._limit_with_rate(
            value=u_t[1],
            prev=u_prev[1],
            mag_min=self.params["thr_min"],
            mag_max=self.params["thr_max"],
            rate=self.params["thr_rate"],
        )
        u_applied = np.array([ele_cmd, thr_cmd])

        # Current state
        x = np.array(self.x_history[-1], dtype=float).reshape(-1)
        u_next = self._integrate(x, u_applied)

        # Bookkeeping
        self.x_history.append(u_next)
        self.u_history.append(u_applied)
        self.time_step += 1

        if self.selected_state_output:
            return np.array(u_next[self.selected_state_index])
        return np.array(u_next)

    # ---------- Internal helpers ----------
    def _limit_with_rate(
        self,
        value: float,
        prev: float,
        mag_min: float,
        mag_max: float,
        rate: float,
    ) -> float:
        """Apply magnitude and symmetric rate limits over one `dt` step."""
        # Magnitude limits first
        value = float(np.clip(value, mag_min, mag_max))
        # Rate constraint
        max_step = float(rate * self.dt)
        value = float(np.clip(value, prev - max_step, prev + max_step))
        # Clip again to magnitude to avoid accumulation errors
        return float(np.clip(value, mag_min, mag_max))

    def _integrate(self, x: np.ndarray, u_applied: np.ndarray):
        """Euler integrate one step of the nonlinear longitudinal dynamics."""
        u = float(x[0])  # forward velocity
        w = float(x[1])  # vertical velocity (body z positive down)
        q = float(x[2])  # pitch rate
        theta = float(x[3])  # pitch angle

        ele = float(u_applied[0])
        throttle = float(u_applied[1])

        # Shorthands
        rho = self.params["rho"]
        S = self.params["S"]
        cbar = self.params["cbar"]
        m = self.params["m"]
        Iyy = self.params["Iyy"]
        g = self.params["g"]

        # Kinematics
        V = float(np.sqrt(max(u * u + w * w, 1e-6)))
        alpha = float(np.arctan2(w, u))
        qhat = float(q * cbar / (2.0 * max(V, 1e-3)))
        q_dyn = 0.5 * rho * V * V

        # Aerodynamic coefficients (lift/drag/moment)
        CL = (
            self.params["CL0"]
            + self.params["CL_alpha"] * alpha
            + self.params["CL_de"] * ele
            + self.params["CL_q"] * qhat
        )
        CD = self.params["CD0"] + self.params["k_ind"] * (CL * CL)
        Cm = (
            self.params["Cm0"]
            + self.params["Cm_alpha"] * alpha
            + self.params["Cm_de"] * ele
            + self.params["Cm_q"] * qhat
        )

        # Aerodynamic forces/moments in body axes (X forward, Z down)
        X_aero = -q_dyn * S * CD
        Z_aero = -q_dyn * S * CL
        My_aero = q_dyn * S * cbar * Cm

        # Simple thrust model along body X
        T0 = self.params["T_static"]
        Vmax = self.params["V_max"]
        thrust = max(0.0, 1.0 - V / max(Vmax, 1e-3)) * T0
        thrust *= np.clip(throttle, 0.0, 1.0)

        X_total = X_aero + thrust
        Z_total = Z_aero

        # Rigid-body longitudinal dynamics (2D); gravity in body axes
        u_dot = -q * w + X_total / m - g * np.sin(theta)
        w_dot = q * u + Z_total / m + g * np.cos(theta)
        q_dot = My_aero / Iyy
        theta_dot = q

        x_dot = np.array([u_dot, w_dot, q_dot, theta_dot], dtype=float)
        return (np.array([u, w, q, theta], dtype=float) + self.dt * x_dot).reshape(-1)
