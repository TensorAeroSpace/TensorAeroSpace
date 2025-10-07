import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class DynamicsNN:
    """Class for working with neural network modeling system dynamics.

    Attributes:
        model: torch.nn.Module
            Neural network for dynamics modeling.
        optimizer: torch.optim.Optimizer
            Optimizer for model training.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize DynamicsNN object.

        Parameters:
            model: torch.nn.Module
                Neural network to be used for dynamics modeling.
        """
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def generate_training_data(
        self,
        num_samples: int = 1000000,
        state_dim: int = 4,
        control_dim: int = 1,
        state_ranges: list[tuple[float, float]] | None = None,
        control_ranges: list[tuple[float, float]] | None = None,
        A: np.ndarray | None = None,
        B: np.ndarray | None = None,
        control_limits: tuple[float, float] = (-25, 25),
        control_signals: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate training data for model.

        Parameters:
            num_samples: int, default 1000000
                Number of samples to generate.
            state_dim: int, default 4
                System state dimension.
            control_dim: int, default 1
                Control input dimension.
            state_ranges: list of tuples
                Value ranges for each state dimension.
            control_ranges: list of tuples, optional
                Value ranges for each control signal.
            A: numpy.ndarray
                System state matrix (dimensions state_dim x state_dim).
            B: numpy.ndarray
                Control matrix (dimensions state_dim x control_dim).
            control_limits: tuple, default (-25, 25)
                Constraints on control signal values.
            control_signals: list of str, optional
                Control signal types for generation.

        Returns:
            tuple of three tensors (states, controls, next_states):
                - states: torch.Tensor
                    System states.
                - controls: torch.Tensor
                    Control inputs.
                - next_states: torch.Tensor
                    Next system states after applying control.

        Exceptions:
            ValueError:
                If input parameters do not meet expected requirements.
        """
        if state_ranges is None:
            raise ValueError("state_ranges must be provided.")
        if A is None:
            raise ValueError("Matrix A must be provided.")
        if B is None:
            raise ValueError("Matrix B must be provided.")

        # Validate input dimensions
        if A.shape != (state_dim, state_dim):
            raise ValueError(
                f"Matrix A should have shape ({state_dim}, {state_dim}), but got {A.shape}"
            )
        if B.shape != (state_dim, control_dim):
            raise ValueError(
                f"Matrix B should have shape ({state_dim}, {control_dim}), but got {B.shape}"
            )
        if len(state_ranges) != state_dim:
            raise ValueError(
                f"State ranges should have length {state_dim}, but got {len(state_ranges)}"
            )

        # Available control signal types
        available_signals = {
            "sine": lambda t, n: torch.sin(t).unsqueeze(-1),
            "sine_07": lambda t, n: (0.7 * torch.sin(2 * t)).unsqueeze(-1),
            "sine_09": lambda t, n: (0.9 * torch.sin(3 * t)).unsqueeze(-1),
            "sine_05_low_freq": lambda t, n: (0.5 * torch.sin(0.5 * t)).unsqueeze(-1),
            "step": lambda t, n, steps=torch.tensor(
                [-1.0, -0.5, 0.0, 0.5, 1.0, -0.7, 0.3, 0.8]
            ): torch.tensor([steps[i % len(steps)] for i in range(n)]).unsqueeze(-1),
            "gaussian_noise": lambda t, n: (torch.randn(n) * 0.5).unsqueeze(-1),
            "uniform_noise": lambda t, n: (torch.rand(n) * 2 - 1).unsqueeze(-1),
            "linear_up": lambda t, n: torch.linspace(-1, 1, n).unsqueeze(-1),
            "linear_down": lambda t, n: torch.linspace(1, -1, n).unsqueeze(-1),
            "sawtooth": lambda t, n: (
                torch.remainder(torch.linspace(0, 4, n), 2) - 1
            ).unsqueeze(-1),
            "damped_sine": lambda t, n: (torch.exp(-t / 5) * torch.sin(t)).unsqueeze(
                -1
            ),
            "chirp": lambda t, n: (torch.sin(t * t / 5)).unsqueeze(-1),
            "modulated_sine": lambda t, n: (
                (1 + 0.5 * torch.sin(0.5 * t)) * torch.sin(2 * t)
            ).unsqueeze(-1),
            "gaussian_pulse": lambda t, n: (
                torch.exp(-((t - 5 * np.pi) ** 2) / (2 * (np.pi) ** 2))
            ).unsqueeze(-1),
            "square_wave": lambda t, n: torch.sign(torch.sin(t)).unsqueeze(-1),
            "triangle_wave": lambda t, n: (
                2 * (torch.remainder(torch.linspace(0, 2, n), 1) - 0.5)
            ).unsqueeze(-1),
            "two_tone": lambda t, n: (
                0.5 * torch.sin(t) + 0.5 * torch.sin(3 * t)
            ).unsqueeze(-1),
            "random_walk": lambda t, n: torch.cumsum(
                torch.randn(n) * 0.1, dim=0
            ).unsqueeze(-1),
            "exponential": lambda t, n: (
                torch.exp(torch.linspace(-2, 2, n)) - torch.exp(torch.tensor(-2.0))
            ).unsqueeze(-1),
            "constant_noise": lambda t, n: (0.2 + torch.randn(n) * 0.05).unsqueeze(-1),
        }

        # Select control signals to use
        if control_signals is None:
            selected_signals = available_signals
        else:
            selected_signals = {}
            for signal_name in control_signals:
                if signal_name in available_signals:
                    selected_signals[signal_name] = available_signals[signal_name]
                else:
                    raise ValueError(f"Invalid control signal type: {signal_name}")

        # Generate diverse states
        states = torch.zeros((num_samples, state_dim))
        for i in range(state_dim):
            states[:, i] = (
                torch.linspace(state_ranges[i][0], state_ranges[i][1], num_samples)
                + torch.randn(num_samples) * 0.1
            )

        # Generate control signals
        controls = torch.zeros((num_samples, control_dim))
        n_signals = len(selected_signals)

        if n_signals > 0:
            n = num_samples // n_signals

            t_global = torch.linspace(0, 10 * np.pi, num_samples)  # Global time vector
            for i, (signal_name, signal_func) in enumerate(selected_signals.items()):
                start_idx = i * n
                end_idx = (
                    (i + 1) * n if i < n_signals - 1 else num_samples
                )  # Ensure all samples are used
                t_local = t_global[start_idx:end_idx]
                n_local = end_idx - start_idx

                # Generate signal and ensure it has the correct shape [n_local, 1]
                signal = signal_func(t_local, n_local)
                if signal.ndim == 1:
                    signal = signal.unsqueeze(
                        -1
                    )  # Add a dimension to make it [n_local, 1]

                controls[start_idx:end_idx, :] = signal

            # Apply control limits
            controls = torch.clamp(
                controls, min=control_limits[0], max=control_limits[1]
            )

        else:
            # Handle case with custom control ranges
            if len(control_ranges) != control_dim:
                raise ValueError(
                    f"Control ranges should have length {control_dim}, but got {len(control_ranges)}"
                )
            for i in range(control_dim):
                controls[:, i] = torch.linspace(
                    control_ranges[i][0], control_ranges[i][1], num_samples
                )
                # Apply control limits
                controls[:, i] = torch.clamp(
                    controls[:, i], min=control_limits[0], max=control_limits[1]
                )

        # Generate output data
        next_states = torch.zeros_like(states)
        for i in range(num_samples):
            next_states[i] = A @ states[i] + B @ controls[i]

        # Shuffle data
        idx = torch.randperm(num_samples)
        return states[idx], controls[idx], next_states[idx]

    def train_and_validate(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        next_states: torch.Tensor,
        epochs: int = 5,
        batch_size: int = 1024,
        val_split: float = 0.2,
        verbose_epoch: int = 10,
    ) -> None:
        """
        Обучение и валидация модели на предоставленных данных.

        Параметры:
            states: torch.Tensor
                Входные состояния системы.
            controls: torch.Tensor
                Управляющие воздействия.
            next_states: torch.Tensor
                Целевые состояния системы.
            epochs: int, по умолчанию 5
                Количество эпох обучения.
            batch_size: int, по умолчанию 1024
                Размер батча для обучения.
            val_split: float, по умолчанию 0.2
                Доля данных для валидации.
            verbose_epoch: int, по умолчанию 10
                Частота вывода информации об обучении.

        Возвращает:
            None

        Сохраняет лучшую модель в файл 'best_model.pth'.
        """
        # Разделение на обучающую и валидационную выборки
        val_size = int(len(states) * val_split)
        train_states, val_states = states[:-val_size], states[-val_size:]
        train_controls, val_controls = controls[:-val_size], controls[-val_size:]
        train_next_states, val_next_states = (
            next_states[:-val_size],
            next_states[-val_size:],
        )
        print("Подготовка данных")
        train_dataset = torch.utils.data.TensorDataset(
            train_states, train_controls, train_next_states
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_states, val_controls, val_next_states
        )
        print("Загрузка  данных")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        print("Начала обучения ")

        for epoch in tqdm(range(epochs)):
            # Обучение
            self.model.train()
            train_loss = 0
            for batch_states, batch_controls, batch_next_states in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(
                    torch.cat([batch_states, batch_controls], dim=-1)
                )
                loss = criterion(predictions, batch_next_states)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Валидация
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_states, batch_controls, batch_next_states in val_loader:
                    predictions = self.model(
                        torch.cat([batch_states, batch_controls], dim=-1)
                    )
                    val_loss += criterion(predictions, batch_next_states).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

            if epoch % verbose_epoch == 0:
                print(
                    f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

    def predict(
        self, state: np.ndarray | torch.Tensor, control: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Предсказание следующего состояния системы на основе текущего состояния и управления.

        Параметры:
            state: array-like или torch.Tensor
                Текущее состояние системы.
            control: array-like или torch.Tensor
                Управляющее воздействие.

        Возвращает:
            numpy.ndarray:
                Предсказанное следующее состояние системы.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            control_tensor = torch.tensor(control, dtype=torch.float32)
            prediction = self.model(torch.cat([state_tensor, control_tensor], dim=-1))
            return prediction.numpy()
