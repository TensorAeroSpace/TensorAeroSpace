import importlib
from abc import ABC

from huggingface_hub import HfApi, snapshot_download


def get_class_from_string(class_path):
    # Разделяем путь на имя модуля и имя класса
    module_name, class_name = class_path.rsplit('.', 1)
    
    # Динамически импортируем модуль
    module = importlib.import_module(module_name)
    
    # Получаем класс из модуля
    cls = getattr(module, class_name)
    
    return cls

class BaseRLModel(ABC):
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
        folder_path = snapshot_download(repo_id=repo_name, token=access_token, revision=version)
        return folder_path


def serialize_env(env):
    # Получаем начальное состояние и ссылку на сигнал из env
    env_data = env.get_init_args()
    env_data['reference_signal'] = env_data['reference_signal'].tolist()
    return env_data


class TheEnvironmentDoesNotMatch(Exception):
    message = "Error The environment does not match the downloaded one"
