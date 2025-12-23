"""PID-based control baselines.

This module provides utilities for running classic PID controllers and logging
their performance in TensorAeroSpace environments.
"""

import datetime
import json
from pathlib import Path

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)


class PID(BaseRLModel):
    """PID controller implementation for control systems.

    This class implements a PID (Proportional-Integral-Derivative) controller
    for automatic control systems. The PID controller uses proportional (P),
    integral (I), and derivative (D) components to compute the control signal.

    Args:
        env: Gymnasium environment. Defaults to None.
        kp (float): Proportional gain. Defaults to 1.
        ki (float): Integral gain. Defaults to 1.
        kd (float): Derivative gain. Defaults to 0.5.
        dt (float): Time step (time difference between consecutive updates). Defaults to 0.01.

    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        dt (float): Time step.
        integral (float): Accumulated integral value.
        prev_error (float): Previous error value for derivative computation.
        env: Gymnasium environment.

    Example:
        >>> pid = PID(env=env, kp=0.1, ki=0.01, kd=0.05, dt=1)
        >>> control_signal = pid.select_action(10, 7)
    """

    def __init__(self, env=None, kp=1, ki=1, kd=0.5, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        self.env = env

    def select_action(self, setpoint, measurement):
        """Compute and return control signal based on setpoint and measurement.

        This method uses the current measurement and setpoint to compute the error,
        then applies the PID algorithm to compute the control signal.

        Args:
            setpoint (float): Desired value that the system should reach.
            measurement (float): Current measured value.

        Returns:
            float: Control signal computed by the PID controller.

        Example:
            >>> pid = PID(env=env, kp=0.1, ki=0.01, kd=0.05, dt=1)
            >>> control_signal = pid.select_action(10, 7)
            >>> print(control_signal)
        """
        error = setpoint - measurement
        self.integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

    def get_param_env(self):
        """Get environment and agent parameters for saving.

        Returns:
            dict: Dictionary with environment and agent policy parameters.
        """
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        print(env_name)
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"
        env_params = {}

        # Добавление информации о пространстве действий и пространстве состояний
        try:
            action_space = str(self.env.action_space)
            env_params["action_space"] = action_space
        except AttributeError:
            pass

        try:
            observation_space = str(self.env.observation_space)
            env_params["observation_space"] = observation_space
        except AttributeError:
            pass

        if "tensoraerospace" in env_name:
            env_params = serialize_env(self.env)

        policy_params = {
            "ki": self.ki,
            "kp": self.kp,
            "kd": self.kd,
            "dt": self.dt,
        }
        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(self, path=None):
        """Save PID model to the specified directory.

        If path is not specified, creates a directory with current date and time.

        Args:
            path (str, optional): Path where the model will be saved. If None,
                creates a directory with current date and time.

        Returns:
            Path: Path to the directory with saved model.
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        # Текущая дата и время в формате 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__
        # Создание пути в текущем каталоге с датой и временем

        save_dir = path / date_str
        config_path = save_dir / "config.json"

        # Создание директории, если она не существует
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохранение модели
        config = self.get_param_env()
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)

        return save_dir

    @classmethod
    def __load(cls, path):
        """Load PID model from the specified directory.

        Args:
            path (str or Path): Path to directory with saved model.

        Returns:
            PID: Loaded PID model instance.

        Raises:
            TheEnvironmentDoesNotMatch: If agent type does not match expected.
        """
        path = Path(path)
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"

        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch
        if "tensoraerospace" in config["env"]["name"]:
            # Десериализуем параметры среды, преобразуя списки в numpy массивы
            env_params = deserialize_env_params(config["env"]["params"])
            env = get_class_from_string(config["env"]["name"])(**env_params)
        else:
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])

        return new_agent

    @classmethod
    def from_pretrained(cls, repo_name, access_token=None, version=None):
        """Load pretrained model from local path or Hugging Face Hub.

        Args:
            repo_name (str): Repository name or local path to model.
            access_token (str, optional): Access token for Hugging Face Hub.
            version (str, optional): Model version to load.

        Returns:
            PID: Loaded PID model instance.
        """
        path = Path(repo_name)
        # Проверяем существование пути (включая относительные пути)
        if path.exists() and path.is_dir():
            new_agent = cls.__load(path)
            return new_agent
        # Проверяем, является ли это локальным путем (начинается с ./ или ../)
        elif (
            repo_name.startswith(("./", "../")) or "/" in repo_name or "\\" in repo_name
        ):
            # Это локальный путь, но директория не существует
            raise FileNotFoundError(f"Локальная директория не найдена: {repo_name}")
        else:
            # Это имя репозитория для Hugging Face Hub
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent
