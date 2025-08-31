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
    """
    Класс PIDController реализует ПИД-регулятор для систем управления.

    Этот класс предназначен для создания и использования ПИД-регулятора в системах
    автоматического управления. ПИД-регулятор использует пропорциональный (P), интегральный (I)
    и дифференциальный (D) компоненты для вычисления управляющего сигнала.

    Атрибуты:
        kp (float): Коэффициент пропорциональной составляющей.
        ki (float): Коэффициент интегральной составляющей.
        kd (float): Коэффициент дифференциальной составляющей.
        dt (float): Шаг времени (разница времени между последовательными обновлениями).
        integral (float): Накопленное значение интегральной составляющей.
        prev_error (float): Предыдущее значение ошибки для вычисления дифференциальной составляющей.

    Методы:
        update(setpoint, measurement): Вычисляет и возвращает управляющий сигнал на основе заданного значения и текущего измерения.

    Args:
        kp (float): Коэффициент пропорциональной составляющей.
        ki (float): Коэффициент интегральной составляющей.
        kd (float): Коэффициент дифференциальной составляющей.
        dt (float): Шаг времени (разница времени между последовательными обновлениями).

    Пример:
        >>> pid = PIDController(0.1, 0.01, 0.05, 1)
        >>> control_signal = pid.update(10, 7)
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
        """
        Вычисляет и возвращает управляющий сигнал на основе заданного значения (setpoint) и текущего измерения.

        Этот метод использует текущее измерение и заданное значение для вычисления ошибки,
        затем применяет ПИД-алгоритм для вычисления управляющего сигнала.

        Args:
            setpoint (float): Заданное значение, к которому должна стремиться система.
            measurement (float): Текущее измеренное значение.

        Returns:
            float: Управляющий сигнал, вычисленный на основе ПИД-регулятора.

        Пример:
            >>> pid = PIDController(0.1, 0.01, 0.05, 1)
            >>> control_signal = pid.update(10, 7)
            >>> print(control_signal)
        """
        error = setpoint - measurement
        self.integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

    def get_param_env(self):
        """Получает параметры среды и агента для сохранения.

        Returns:
            dict: Словарь с параметрами среды и политики агента.
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
        """Сохраняет модель PID в указанной директории.

        Если путь не указан, создает директорию с текущей датой и временем.

        Args:
            path (str, optional): Путь, где будет сохранена модель. Если None,
                                создается директория с текущей датой и временем.
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        # Текущая дата и время в формате 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__
        # Создание пути в текущем каталоге с датой и временем

        config_path = path / date_str / "config.json"

        # Создание директории, если она не существует
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохранение модели
        config = self.get_param_env()
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)

    @classmethod
    def __load(cls, path):
        """Загружает модель PID из указанной директории.

        Args:
            path (str or Path): Путь к директории с сохраненной моделью.

        Returns:
            PID: Загруженный экземпляр модели PID.

        Raises:
            TheEnvironmentDoesNotMatch: Если тип агента не соответствует ожидаемому.
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
        """Загружает предобученную модель из локального пути или Hugging Face Hub.

        Args:
            repo_name (str): Имя репозитория или локальный путь к модели.
            access_token (str, optional): Токен доступа для Hugging Face Hub.
            version (str, optional): Версия модели для загрузки.

        Returns:
            PID: Загруженный экземпляр модели PID.
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
