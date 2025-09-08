"""
Базовые классы и утилиты для агентов обучения с подкреплением.

Этот модуль содержит базовый абстрактный класс BaseRLModel, который определяет
общий интерфейс для всех алгоритмов обучения с подкреплением в библиотеке
TensorAeroSpace. Также включает утилиты для работы с моделями, их сериализации
и интеграции с Hugging Face Hub.

Основные компоненты:
- BaseRLModel: Базовый класс для всех RL-алгоритмов
- get_class_from_string: Утилита для динамического импорта классов
- serialize_env: Функция для сериализации сред
- TheEnvironmentDoesNotMatch: Исключение для несоответствия сред
"""

import importlib
from abc import ABC

from huggingface_hub import HfApi, snapshot_download


def get_class_from_string(class_path):
    """Динамически импортирует и возвращает класс по строковому пути.

    Args:
        class_path (str): Полный путь к классу в формате 'module.submodule.ClassName'.

    Returns:
        type: Класс, соответствующий указанному пути.

    Raises:
        ImportError: Если модуль не может быть импортирован.
        AttributeError: Если класс не найден в модуле.
    """
    # Разделяем путь на имя модуля и имя класса
    module_name, class_name = class_path.rsplit(".", 1)

    # Динамически импортируем модуль
    module = importlib.import_module(module_name)

    # Получаем класс из модуля
    cls = getattr(module, class_name)

    return cls


class BaseRLModel(ABC):
    """Базовый абстрактный класс для моделей обучения с подкреплением.

    Этот класс определяет общий интерфейс для всех алгоритмов обучения с подкреплением
    в библиотеке TensorAeroSpace. Все конкретные реализации алгоритмов должны наследоваться
    от этого класса и реализовывать его абстрактные методы.

    Attributes:
        Базовый класс не содержит специфических атрибутов.
    """

    def __init__(self) -> None:
        """Инициализирует объект класса BaseRLModel."""
        super().__init__()

    def get_env(self):
        """Возвращает текущую среду обучения модели.

        Returns:
            object: Объект среды, используемой для обучения модели.
        """
        pass

    def train(self):
        """Запускает процесс обучения модели."""
        pass

    def action_probability(self):
        """Возвращает вероятности действий для последнего состояния.

        Returns:
            list: Список вероятностей действий.
        """
        pass

    def save(self):
        """Сохраняет текущую модель в файл."""
        pass

    def load(self):
        """Загружает модель из файла."""
        pass

    def predict(self):
        """Делает прогноз на основе входных данных.

        Returns:
            Any: Результат прогнозирования.
        """
        pass

    def get_param_env(self):
        """Получает параметры текущей среды.

        Returns:
            dict: Словарь параметров среды.
        """
        pass

    def publish_to_hub(self, repo_name, folder_path, access_token=None):
        """Публикует модель в Hugging Face Hub.

        Args:
            repo_name (str): Название репозитория в Hub.
            folder_path (str): Путь к папке с моделью.
            access_token (str, optional): Токен доступа для аутентификации.
        """
        api = HfApi()
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )

    @classmethod
    def from_pretrained(cls, repo_name, access_token=None, version=None):
        """Загружает предобученную модель из Hugging Face Hub.

        Args:
            repo_name (str): Название репозитория в Hub.
            access_token (str, optional): Токен доступа для аутентификации.
            version (str, optional): Версия модели для загрузки.

        Returns:
            str: Путь к загруженной папке с моделью.
        """
        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return folder_path


def serialize_env(env):
    """Сериализует объект среды в словарь для сохранения.

    Args:
        env: Объект среды, который нужно сериализовать.

    Returns:
        dict: Словарь с параметрами среды, включая все numpy массивы в виде списков.
    """
    import numpy as np

    # Получаем начальное состояние и ссылку на сигнал из env
    env_data = env.get_init_args()

    # Рекурсивно преобразуем все numpy массивы в списки
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    return convert_numpy_to_list(env_data)


def deserialize_env_params(env_params):
    """Десериализует параметры среды, преобразуя списки обратно в numpy массивы.

    Args:
        env_params (dict): Словарь с параметрами среды.

    Returns:
        dict: Словарь с параметрами среды, где списки преобразованы в numpy массивы.
    """
    import numpy as np

    # Рекурсивно преобразуем списки в numpy массивы для известных параметров
    def convert_list_to_numpy(obj, key=None):
        if isinstance(obj, list) and key in [
            "reference_signal",
            "initial_state",
            "alpha_states",
        ]:
            return np.array(obj)
        elif isinstance(obj, dict):
            return {k: convert_list_to_numpy(v, k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_list_to_numpy(item) for item in obj]
        else:
            return obj

    return convert_list_to_numpy(env_params)


class TheEnvironmentDoesNotMatch(Exception):
    """Исключение, возникающее при несоответствии загруженной среды ожидаемой.

    Это исключение выбрасывается, когда загруженная из файла среда не соответствует
    той, которая ожидается для работы с моделью.

    Attributes:
        message (str): Сообщение об ошибке.
    """

    message = "Error The environment does not match the downloaded one"
