"""Incremental model component for IHDP.

This module contains the incremental model used for online system
identification within the IHDP algorithm.
"""

import numpy as np


class IncrementalModel:
    """Provides IncrementalModel class for system identification.

    IncrementalModel computes A and x matrices needed for system identification,
    computes F and G matrices needed for incremental model, and evaluates
    identified model to provide state estimates at next time step.

    Args:
        selected_states: Selected states.
        selected_input: Selected control signals.
        number_time_steps: Number of time steps.
        discretisation_time (float, optional): Discretization time. Defaults to 0.5.
        input_magnitude_limits (int, optional): Input control signal limits. Defaults to 25.
        input_rate_limits (int, optional): Control signal rate constraints. Defaults to 60.
    """

    def __init__(
        self,
        selected_states: list[str],
        selected_input: list[str],
        number_time_steps: int,
        discretisation_time: float = 0.5,
        input_magnitude_limits: float = 25,
        input_rate_limits: float = 60,
    ) -> None:
        """Initialize incremental model buffers and limits.

        Args:
            selected_states: Names of states.
            selected_input: Names of control inputs.
            number_time_steps: Horizon length.
            discretisation_time: Sampling period.
            input_magnitude_limits: Max control magnitude.
            input_rate_limits: Max control rate change.
        """
        # Define the inputs to the incremental model
        self.number_time_steps = number_time_steps
        self.number_states = len(selected_states)
        self.number_inputs = len(selected_input)

        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt = np.zeros((self.number_states, 1))
        self.ut_1 = np.zeros((self.number_inputs, 1))
        self.ut = np.zeros((self.number_inputs, 1))
        self.delta_xt = np.zeros((self.number_states, 1))
        self.delta_ut = np.zeros((self.number_inputs, 1))
        self.xt1_est = np.zeros((self.number_states, 1))

        # Define the data window size
        self.L = 2 * (self.number_inputs + self.number_states)
        self.store_delta_xt = np.zeros((self.number_states, self.number_time_steps))
        self.store_delta_xt_0 = np.random.rand(self.number_states, self.L)
        self.store_delta_ut = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_delta_ut_0 = np.random.rand(self.number_inputs, self.L)
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))

        # Define the system identification matrices
        self.F = np.zeros((self.number_states, self.number_states))
        self.G = np.zeros((self.number_states, self.number_inputs))

        # Define the time variable
        self.time_step = 0
        self.discretisation_time = discretisation_time

        # Limitations of the system
        self.input_magnitude_limits = self._as_input_column(
            input_magnitude_limits, "input_magnitude_limits"
        )
        self.input_rate_limits = self._as_input_column(
            input_rate_limits, "input_rate_limits"
        )

    def _as_input_column(self, value, name: str) -> np.ndarray:
        """Return a ``(number_inputs, 1)`` vector, broadcasting scalars."""
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full(self.number_inputs, float(arr[0]), dtype=float)
        elif arr.size != self.number_inputs:
            raise ValueError(
                f"{name} must be scalar or have {self.number_inputs} elements; "
                f"got {arr.size}."
            )
        return arr.reshape(self.number_inputs, 1)

    def _as_input_signal(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[1] != 1:
            arr = arr.reshape(-1, 1)
        if arr.shape != (self.number_inputs, 1):
            raise ValueError(
                f"input has shape {arr.shape}, but incremental model expects "
                f"({self.number_inputs}, 1)."
            )
        return arr

    def _apply_input_limits(self, value: np.ndarray) -> np.ndarray:
        ut_0 = self._as_input_signal(value)
        rate_limited = np.clip(
            ut_0,
            self.ut_1 - self.input_rate_limits * self.discretisation_time,
            self.ut_1 + self.input_rate_limits * self.discretisation_time,
        )
        return np.clip(
            rate_limited,
            -self.input_magnitude_limits,
            self.input_magnitude_limits,
        )

    def save_matrix(self):
        """Save identification matrices to disk (NumPy .npy files)."""
        np.save("./incremental_model/g", self.G)
        np.save("./incremental_model/f", self.F)
        np.save("./incremental_model/delta_ut", self.delta_ut)
        np.save("./incremental_model/delta_xt", self.delta_xt)

    def load_matrix(self):
        """Load identification matrices from disk (NumPy .npy files)."""
        self.G = np.load(
            "./incremental_model/g.npy",
        )
        self.F = np.load(
            "./incremental_model/f.npy",
        )
        self.delta_ut = np.load(
            "./incremental_model/delta_ut.npy",
        )
        self.delta_xt = np.load(
            "./incremental_model/delta_xt.npy",
        )

    def build_A_LS_matrix(self) -> np.ndarray:
        """Build the least-squares A matrix used for online identification."""

        if self.time_step >= self.L:
            x_component = np.flip(
                self.store_delta_xt[:, self.time_step - self.L : self.time_step], 1
            ).T
            u_component = np.flip(
                self.store_delta_ut[:, self.time_step - self.L : self.time_step], 1
            ).T
        else:
            x_component_1 = np.flip(self.store_delta_xt[:, : self.time_step], 1).T
            x_component_2 = self.store_delta_xt_0[:, : self.L - self.time_step].T
            x_component = np.vstack((x_component_1, x_component_2))

            u_component_1 = np.flip(self.store_delta_ut[:, : self.time_step], 1).T
            u_component_2 = self.store_delta_ut_0[:, : self.L - self.time_step].T
            u_component = np.vstack((u_component_1, u_component_2))
        A_LS_matrix = np.hstack((x_component, u_component))
        return A_LS_matrix

    def build_x_LS_vector(self) -> np.ndarray:
        """Build the least-squares x vector used for online identification."""
        if self.time_step == 0:
            self.xt_1 = self.xt

        # Computation and storage of the gradients
        self.delta_xt = self.xt - self.xt_1
        self.delta_ut = self.ut - self.ut_1
        self.store_delta_xt[:, self.time_step] = np.reshape(
            self.delta_xt, [self.delta_xt.shape[0]]
        )
        self.store_delta_ut[:, self.time_step] = np.reshape(
            self.delta_ut, [self.delta_ut.shape[0]]
        )
        if self.time_step >= self.L:
            x_LS_vector = np.flip(
                self.store_delta_xt[
                    :, self.time_step - self.L + 1 : self.time_step + 1
                ],
                1,
            ).T
        else:
            x_component_1 = np.flip(self.store_delta_xt[:, : self.time_step + 1], 1).T
            x_component_2 = self.store_delta_xt_0[:, : self.L - self.time_step - 1].T
            x_LS_vector = np.vstack((x_component_1, x_component_2))

        return x_LS_vector

    def identify_incremental_model_LS(
        self, xt: np.ndarray, ut_0: np.ndarray
    ) -> np.ndarray:
        """Estimate F and G matrices for the incremental model (least squares)."""
        # Normalize state shape to a column vector and validate dimensionality early.
        # This prevents hard-to-debug NumPy broadcasting errors later when storing
        # state deltas into fixed-size buffers.
        xt_arr = np.asarray(xt)
        if xt_arr.ndim == 1:
            xt_arr = xt_arr.reshape([-1, 1])
        elif xt_arr.ndim == 2 and xt_arr.shape[1] != 1:
            xt_arr = xt_arr.reshape([-1, 1])

        if xt_arr.shape[0] != self.number_states:
            raise ValueError(
                f"xt has shape {xt_arr.shape}, but incremental model expects "
                f"({self.number_states}, 1)."
            )

        ut_0 = self._as_input_signal(ut_0)

        # Verifying that the inputs meets the platforms constraints
        if self.time_step == 0:
            self.ut_1 = ut_0
        ut = self._apply_input_limits(ut_0)

        # Store the input variables
        self.xt = xt_arr
        self.ut = ut
        self.store_input[:, self.time_step] = np.reshape(ut, [ut.shape[0]])

        # Obtain the A matrix and the x vector
        A_LS_matrix = self.build_A_LS_matrix()
        x_LS_vector = self.build_x_LS_vector()
        identified_matrices = np.matmul(
            np.matmul(
                np.linalg.pinv(np.matmul(A_LS_matrix.T, A_LS_matrix)), A_LS_matrix.T
            ),
            x_LS_vector,
        ).T
        self.F = identified_matrices[:, : self.number_states]
        self.G = identified_matrices[:, self.number_states :]

        return self.G

    def evaluate_incremental_model(self, *args: np.ndarray) -> np.ndarray:
        """Estimate states for the next time step.

        Returns:
            xt1_est (_type_): Estimated state for the next time step.
        """

        if len(args) == 0:
            # Estimate the next time step states
            self.xt1_est = (
                self.xt
                + np.matmul(self.F, self.delta_xt)
                + np.matmul(self.G, self.delta_ut)
            )
            return np.asarray(self.xt1_est)
        elif len(args) == 1:
            # Estimate the next time step states
            ut_0 = args[0]
            ut = self._apply_input_limits(ut_0)

            delta_ut = ut - self.ut_1
            xt1_est = (
                self.xt + np.matmul(self.F, self.delta_xt) + np.matmul(self.G, delta_ut)
            )
            return np.asarray(xt1_est)
        elif len(args) == 2:
            self.xt = np.asarray(args[0], dtype=float).reshape(-1, 1)
            ut_0 = self._as_input_signal(args[1])
            if self.time_step == 0:
                self.ut_1 = ut_0
                self.xt_1 = self.xt
            # Estimate the next time step states

            ut = self._apply_input_limits(ut_0)

            self.delta_ut = ut - self.ut_1
            self.delta_xt = self.xt - self.xt_1
            xt1_est = (
                self.xt
                + np.matmul(self.F, self.delta_xt)
                + np.matmul(self.G, self.delta_ut)
            )
            return np.asarray(xt1_est)

        raise ValueError("Unexpected number of arguments.")

    def update_incremental_model_attributes(self) -> None:
        """Update attributes that change with each time step."""

        # Update the object state and input variables
        self.xt_1 = self.xt
        self.ut_1 = self.ut
        self.time_step += 1

    def restart_time_step(self) -> None:
        """Reset time step to zero."""
        self.time_step = 0

    def restart_incremental_model(self) -> None:
        """Restart the incremental model."""
        self.time_step = 0
        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt = np.zeros((self.number_states, 1))
        self.ut_1 = np.zeros((self.number_inputs, 1))
        self.ut = np.zeros((self.number_inputs, 1))
        self.delta_xt = np.zeros((self.number_states, 1))
        self.delta_ut = np.zeros((self.number_inputs, 1))
        self.xt1_est = np.zeros((self.number_states, 1))
        self.store_delta_xt = np.zeros((self.number_states, self.number_time_steps))
        self.store_delta_ut = np.zeros((self.number_inputs, self.number_time_steps))
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))
