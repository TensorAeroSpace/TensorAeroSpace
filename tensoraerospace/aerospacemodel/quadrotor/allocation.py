"""Motor allocation / mixing for X-configuration quadrotors.

Maps the 4 rotor speeds-squared :math:`(\\omega_1^2, \\omega_2^2, \\omega_3^2,
\\omega_4^2)` (the "rotor level") to / from the virtual control vector
:math:`(T, \\tau_x, \\tau_y, \\tau_z)` (collective thrust + body torques)
that the ODE consumes.

PX4 X-configuration motor convention (looking down on the body, x forward):

.. code:: text

       x (front)
       ^
       |
   M3  |  M1   (M1, M2 spin CCW; M3, M4 spin CW)
     \\ | /
      \\|/
   ----+----> y (right)
      /|\\
     / | \\
   M2  |  M4

With the symmetric X-frame, each motor sits at distance
``arm_length / sqrt(2)`` from both the body x and y axes — that's the
effective lever arm for roll and pitch torques.

References:
- PX4 Airframe Reference, "Quadrotor X" entry.
- Lu et al. 2019, *Quadrotor Fault Tolerant Incremental Sliding Mode
  Control* (Aerospace Sci & Tech) — uses exactly this mixing.
- Wang et al. 2019, *Quadrotor fault-tolerant incremental nonsingular
  terminal sliding mode control*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XConfigAllocator:
    """Bidirectional X-quad rotor↔virtual allocator.

    The allocation matrix :math:`M` is defined by

    .. math::

        \\begin{bmatrix} T \\\\ \\tau_x \\\\ \\tau_y \\\\ \\tau_z \\end{bmatrix}
        = M \\begin{bmatrix} \\omega_1^2 \\\\ \\omega_2^2 \\\\ \\omega_3^2 \\\\ \\omega_4^2 \\end{bmatrix},

    where (with :math:`a = L/\\sqrt 2`)

    .. math::

        M = \\begin{bmatrix}
              k_T   &  k_T  &  k_T  &  k_T   \\\\
             -k_T a &  k_T a&  k_T a& -k_T a \\\\
              k_T a & -k_T a&  k_T a& -k_T a \\\\
              k_M   &  k_M  & -k_M  & -k_M
            \\end{bmatrix}.

    The matrix is full-rank for non-degenerate parameters, so the
    inverse :func:`unmix` is well-defined.

    Args:
        k_T: thrust coefficient (N per :math:`(\\mathrm{rad/s})^2`).
            Default value matches a typical 1.5 kg quad with
            :math:`\\omega_\\max \\approx 1000\\,\\mathrm{rad/s}` and
            per-motor max thrust :math:`\\approx 7.5\\,\\mathrm{N}`.
        k_M: yaw-torque coefficient (N·m per :math:`(\\mathrm{rad/s})^2`).
            Typically ``k_M / k_T ≈ 0.016 m`` for small props.
        arm_length: distance from CG to each rotor, m.
    """

    k_T: float = 7.5e-6
    k_M: float = 1.2e-7   # ≈ 0.016 · k_T  (yaw moment per unit thrust)
    arm_length: float = 0.225

    def __post_init__(self) -> None:
        if self.k_T <= 0:
            raise ValueError(f"k_T must be positive; got {self.k_T}")
        if self.k_M <= 0:
            raise ValueError(f"k_M must be positive; got {self.k_M}")
        if self.arm_length <= 0:
            raise ValueError(f"arm_length must be positive; got {self.arm_length}")
        a = self.arm_length / np.sqrt(2.0)
        kT, kM = self.k_T, self.k_M
        self._M = np.array(
            [
                [ kT,     kT,    kT,     kT  ],
                [-kT * a, kT * a, kT * a, -kT * a],
                [ kT * a, -kT * a, kT * a, -kT * a],
                [ kM,     kM,   -kM,    -kM  ],
            ],
            dtype=np.float64,
        )
        self._M_inv = np.linalg.inv(self._M)

    # ---- forward: rotor speeds² → virtual ------------------------------

    def mix(self, omega_squared: np.ndarray) -> np.ndarray:
        """Map :math:`(\\omega_1^2, \\ldots, \\omega_4^2)` → :math:`(T, \\tau_x, \\tau_y, \\tau_z)`.

        Args:
            omega_squared: shape ``(4,)``, units :math:`(\\mathrm{rad/s})^2`.
                Must be non-negative element-wise.

        Returns:
            Virtual control vector ``[T, τ_x, τ_y, τ_z]`` of shape ``(4,)``.
        """
        omega_squared = np.asarray(omega_squared, dtype=np.float64).reshape(-1)
        if omega_squared.size != 4:
            raise ValueError(
                f"omega_squared must have 4 elements; got {omega_squared.size}"
            )
        if np.any(omega_squared < 0):
            raise ValueError("omega_squared must be non-negative")
        return self._M @ omega_squared

    # ---- inverse: virtual → rotor speeds² ------------------------------

    def unmix(self, virtual: np.ndarray) -> np.ndarray:
        """Map :math:`(T, \\tau_x, \\tau_y, \\tau_z)` → :math:`(\\omega_1^2, \\ldots, \\omega_4^2)`.

        Note: the result may contain **negative** entries when the
        commanded virtual triple is not realisable by 4 unidirectional
        rotors (e.g. requesting :math:`T = 0` with non-zero pitch torque).
        The caller should pass the result through :func:`saturate` before
        feeding it into :func:`mix`.

        Args:
            virtual: shape ``(4,)``, units N (T) and N·m (τ).

        Returns:
            Rotor speeds-squared ``[ω₁², ω₂², ω₃², ω₄²]`` of shape ``(4,)``.
            Possibly negative — pass through :func:`saturate` if needed.
        """
        virtual = np.asarray(virtual, dtype=np.float64).reshape(-1)
        if virtual.size != 4:
            raise ValueError(f"virtual must have 4 elements; got {virtual.size}")
        return self._M_inv @ virtual

    # ---- saturation ----------------------------------------------------

    def saturate(
        self,
        omega_squared: np.ndarray,
        omega_min: float = 0.0,
        omega_max: float = 1000.0,
    ) -> np.ndarray:
        """Clip rotor speeds-squared into :math:`[\\omega_\\min^2, \\omega_\\max^2]`.

        Element-wise clip — does **not** preserve any virtual quantity.
        For thrust-priority allocation strategies, see the literature
        (e.g. Faessler 2017); this method is the simplest sane default.

        Args:
            omega_squared: shape ``(4,)``, possibly out-of-bounds.
            omega_min: lower rotor-speed bound, rad/s. Default 0
                (perpetual zero allowed — useful for rotor-loss
                scenarios). Must be non-negative.
            omega_max: upper rotor-speed bound, rad/s. Default 1000
                (≈ 9550 RPM).

        Returns:
            Clipped ``[ω₁², ω₂², ω₃², ω₄²]``.
        """
        omega_squared = np.asarray(omega_squared, dtype=np.float64).reshape(-1)
        if omega_squared.size != 4:
            raise ValueError(
                f"omega_squared must have 4 elements; got {omega_squared.size}"
            )
        if omega_min < 0:
            raise ValueError(f"omega_min must be non-negative; got {omega_min}")
        if omega_max <= omega_min:
            raise ValueError(
                f"omega_max must be > omega_min; got {omega_max} <= {omega_min}"
            )
        lo = float(omega_min) ** 2
        hi = float(omega_max) ** 2
        return np.clip(omega_squared, lo, hi)

    # ---- introspection -------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """Read-only 4×4 forward allocation matrix M."""
        return self._M.copy()

    @property
    def matrix_inverse(self) -> np.ndarray:
        """Read-only 4×4 inverse matrix M⁻¹."""
        return self._M_inv.copy()


def default_allocator() -> XConfigAllocator:
    """Allocator matching the default `QuadrotorParameters` (1.5 kg X-frame)."""
    return XConfigAllocator()
