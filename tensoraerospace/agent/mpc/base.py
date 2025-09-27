from typing import Callable, Dict, Tuple

import numpy as np
import torch


class AircraftMPC:
    """Class for implementing Model Predictive Control (MPC) for aircraft systems.

    Attributes:
        dynamics_model: System dynamics model.
        horizon (int): Prediction horizon.
        dt (float): Discretization step.
        weights (dict): Weights for objective function, including:
            - 'theta_tracking': Tracking error weight.
            - 'control_effort': Control effort weight.
            - 'delta_control': Control change weight.
        state_dim (int): State vector dimension.
        control_dim (int): Control vector dimension.
        u_max (float): Maximum control value.
        delta_u_max (float): Maximum control change between steps.
        learning_rate (float): Learning rate for gradient descent.
        penalty_weight (float): Penalty function weight for constraints.
        iterations (int): Number of optimization iterations.
        increment (float): Small increment for numerical gradient computation.
    """

    def __init__(
        self,
        dynamics_model: Callable[[torch.Tensor], torch.Tensor],
        horizon: int = 2,
        dt: float = 0.1,
        weights: Dict[str, float] = {
            "theta_tracking": 10000.0,
            "control_effort": 0.1,
            "delta_control": 0.01,
        },
        state_dim: int = 4,
        control_dim: int = 1,
        u_max: float = 10.0,
        delta_u_max: float = 0.001,
        learning_rate: float = 10e-5,
        penalty_weight: float = 1_000,
        iterations: int = 150,
        increment: float = 1e-3,
    ) -> None:
        """Initialize MPC controller.

        Args:
            dynamics_model: System dynamics model.
            horizon (int): Prediction horizon.
            dt (float): Discretization step.
            weights (dict): Objective function weights.
            state_dim (int): System state dimension.
            control_dim (int): System control dimension.
            u_max (float): Maximum control value.
            delta_u_max (float): Maximum control change between steps.
            learning_rate (float): Learning rate for gradient descent.
            penalty_weight (float): Penalty function weight for constraints.
            iterations (int): Number of optimization iterations.
            increment (float): Small increment for numerical gradient computation.
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.dt = dt
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.weights = weights
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.learning_rate = learning_rate  # БЫЛО 0.1
        self.penalty_weight = penalty_weight  # Вес штрафной функции
        self.iterations = iterations
        self.increment = increment

    def cost_function(
        self, X: np.ndarray, U: np.ndarray, theta_ref_np: np.ndarray
    ) -> float:
        """
        Вычисляет значение целевой функции.

        Args:
            X: Массив состояний (horizon+1, state_dim)
            U: Массив управлений (horizon, control_dim)
            theta_ref_np: Опорная траектория (horizon+1,)

        Returns:
            Значение целевой функции.
        """
        cost = 0
        # Ошибка отслеживания
        cost += self.weights["theta_tracking"] * np.sum((X[:, 3] - theta_ref_np) ** 2)
        # Штраф за управление
        cost += self.weights["control_effort"] * np.sum(U**2)
        # Штраф за изменение управления
        cost += self.weights["delta_control"] * np.sum(np.diff(U, axis=0) ** 2)

        return float(cost)

    def penalty_function(self, U: np.ndarray) -> float:
        """
        Вычисляет значение штрафной функции для ограничений.

        Args:
            U: Массив управлений (horizon, control_dim)

        Returns:
            Значение штрафной функции.
        """
        penalty = 0
        # Ограничения на управление
        penalty += np.sum(np.maximum(0, np.abs(U) - self.u_max) ** 2)
        # Ограничения на изменение управления
        penalty += np.sum(
            np.maximum(0, np.abs(np.diff(U, axis=0)) - self.delta_u_max) ** 2
        )
        return float(penalty)

    def total_cost(
        self, U: np.ndarray, x0: np.ndarray, theta_ref_np: np.ndarray
    ) -> float:
        """
        Вычисляет полное значение целевой функции с учётом штрафов.

        Args:
            U: Массив управлений (horizon, control_dim) - векторизованное представление
            x0: Начальное состояние
            theta_ref_np: Опорная траектория

        Returns:
            Полное значение целевой функции.
        """
        U_reshaped = U.reshape((self.horizon, self.control_dim))
        X = self.predict_trajectory(x0, U_reshaped)
        return self.cost_function(
            X, U_reshaped, theta_ref_np
        ) + self.penalty_weight * self.penalty_function(U_reshaped)

    def predict_trajectory(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Прогнозирует траекторию состояний на основе модели динамики.

        Args:
            x0: Начальное состояние
            U: Массив управлений (horizon, control_dim)

        Returns:
            Массив состояний (horizon+1, state_dim)
        """
        X = np.zeros((self.horizon + 1, self.state_dim))
        X[0] = x0
        for t in range(self.horizon):
            X[t + 1] = (
                self.dynamics_model(
                    torch.cat(
                        [
                            torch.tensor(X[t], dtype=torch.float32).unsqueeze(0),
                            torch.tensor(U[t], dtype=torch.float32).unsqueeze(0),
                        ],
                        dim=-1,
                    )
                )
                .detach()
                .numpy()
                .flatten()
            )
        return X

    def optimize_control(
        self, x0: np.ndarray, theta_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оптимизирует последовательность управления с использованием градиентного спуска.

        Args:
            x0: Начальное состояние системы размерности state_dim.
            theta_ref: Опорная траектория размерности horizon+1.

        Returns:
            tuple:
                - np.ndarray: Оптимальное управление на первом шаге (control_dim,).
                - np.ndarray: Прогнозируемая траектория состояний размерности (horizon, state_dim).
        """
        theta_ref_np = np.array(theta_ref, dtype=np.float32)

        # Начальное приближение управления (например, нули)
        U = np.zeros((self.horizon * self.control_dim))

        for _ in range(self.iterations):
            # Вычисляем градиент целевой функции по U численно (конечные разности)
            grad = np.zeros_like(U)
            for i in range(len(U)):
                U_plus = U.copy()
                U_plus[i] += self.increment  # Малое приращение
                grad[i] = (
                    self.total_cost(U_plus, x0, theta_ref_np)
                    - self.total_cost(U, x0, theta_ref_np)
                ) / self.increment

            # Обновляем управление с помощью градиентного спуска
            U -= self.learning_rate * grad

            # Проецируем U на допустимое множество (не обязательно, если используется штрафная функция)
            U = np.clip(
                U.reshape((self.horizon, self.control_dim)), -self.u_max, self.u_max
            ).flatten()
            for t in range(self.horizon - 1):
                U[t + 1] = np.clip(
                    U[t + 1], U[t] - self.delta_u_max, U[t] + self.delta_u_max
                )

        # Возвращаем первое управление из оптимальной последовательности
        return (
            U.reshape((self.horizon, self.control_dim))[0],
            self.predict_trajectory(x0, U.reshape((self.horizon, self.control_dim)))[
                1:
            ],
        )
