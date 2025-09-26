"""Base classes and utilities for reinforcement learning agents.

This module contains the base abstract class BaseRLModel, which defines
a common interface for all reinforcement learning algorithms in the
TensorAeroSpace library. Also includes utilities for working with models,
their serialization and integration with Hugging Face Hub.

Main components:
    - BaseRLModel: Base class for all RL algorithms
    - get_class_from_string: Utility for dynamic class import
    - serialize_env: Function for environment serialization
    - TheEnvironmentDoesNotMatch: Exception for environment mismatch
"""

import importlib
from abc import ABC

from huggingface_hub import HfApi, snapshot_download


def get_class_from_string(class_path):
    """Dynamically imports and returns a class by string path.

    Args:
        class_path (str): Full path to class in format 'module.submodule.ClassName'.

    Returns:
        type: Class corresponding to the specified path.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    # Разделяем путь на имя модуля и имя класса
    module_name, class_name = class_path.rsplit(".", 1)

    # Динамически импортируем модуль
    module = importlib.import_module(module_name)

    # Получаем класс из модуля
    cls = getattr(module, class_name)

    return cls


class BaseRLModel(ABC):
    """Base abstract class for reinforcement learning models.

    This class defines a common interface for all reinforcement learning algorithms
    in the TensorAeroSpace library. All concrete algorithm implementations should inherit
    from this class and implement its abstract methods.

    Attributes:
        Base class contains no specific attributes.
    """

    def __init__(self) -> None:
        """Initialize BaseRLModel object."""
        super().__init__()

    def get_env(self):
        """Returns the current training environment of the model.

        Returns:
            object: Environment object used for model training.
        """
        pass

    def train(self):
        """Start the model training process."""
        pass

    def action_probability(self):
        """Returns action probabilities for the last state.

        Returns:
            list: List of action probabilities.
        """
        pass

    def save(self):
        """Save current model to file."""
        pass

    def load(self):
        """Load model from file."""
        pass

    def predict(self):
        """Make prediction based on input data.

        Returns:
            Any: Prediction result.
        """
        pass

    def get_param_env(self):
        """Get parameters of the current environment.

        Returns:
            dict: Dictionary of environment parameters.
        """
        pass

    def publish_to_hub(self, repo_name, folder_path, access_token=None):
        """Publish model to Hugging Face Hub.

        Args:
            repo_name (str): Repository name in Hub.
            folder_path (str): Path to model folder.
            access_token (str, optional): Access token for authentication.
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
        """Load pretrained model from Hugging Face Hub.

        Args:
            repo_name (str): Repository name in Hub.
            access_token (str, optional): Access token for authentication.
            version (str, optional): Model version to load.

        Returns:
            str: Path to downloaded model folder.
        """
        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return folder_path


def serialize_env(env):
    """Serialize environment object to dictionary for saving.

    Args:
        env: Environment object to serialize.

    Returns:
        dict: Dictionary with environment parameters, including all numpy arrays as lists.
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
    """Deserialize environment parameters, converting lists back to numpy arrays.

    Args:
        env_params (dict): Dictionary with environment parameters.

    Returns:
        dict: Dictionary with environment parameters where lists are converted to numpy arrays.
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
    """Exception raised when loaded environment does not match expected one.

    This exception is raised when the environment loaded from file does not match
    the one expected for working with the model.

    Attributes:
        message (str): Error message.
    """

    message = "Error The environment does not match the downloaded one"
