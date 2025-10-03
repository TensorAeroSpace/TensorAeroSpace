import os
import pickle
import random
from typing import List, Tuple, Union

import numpy as np


class ReplayMemory:
    """Хранилище повторных сэмплов для алгоритмов обучения с подкреплением.

    Args:
        capacity (int): Максимальная вместимость хранилища.
        seed (int): Зерно для генерации случайных чисел.

    Attributes:
        capacity (int): Максимальная вместимость хранилища.
        buffer (List): Буфер для хранения повторных сэмплов.
        position (int): Текущая позиция в буфере.

    """

    def __init__(self, capacity: int, seed: int):
        random.seed(seed)
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position: int = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        next_state: np.ndarray,
        done: Union[bool, float],
    ) -> None:
        """Добавление повторного сэмпла в хранилище.

        Args:
            state: Входное состояние.
            action: Действие.
            reward: Награда.
            next_state: Следующее состояние.
            done: Маска окончания эпизода (True/False или 0.0/1.0).

        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Сэмплирование пакета повторных сэмплов из хранилища.

        Args:
            batch_size (int): Размер пакета.

        Returns:
            Tuple: Кортеж с состояниями, действиями, наградами,
            следующими состояниями и флагами окончания.

        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        """Возвращает текущий размер хранилища.

        Returns:
            int: Размер хранилища.

        """
        return len(self.buffer)

    def save_buffer(
        self, env_name: str, suffix: str = "", save_path: str | None = None
    ) -> None:
        """Сохранение буфера на диск.

        Args:
            env_name (str): Название окружения.
            suffix (str): Суффикс для имени файла. По умолчанию "".
            save_path (str): Путь для сохранения файла. По умолчанию None.

        """
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print("Saving buffer to {}".format(save_path))

        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path: str) -> None:
        """Загрузка буфера из файла.

        Args:
            save_path (str): Путь к файлу для загрузки буфера.

        """
        print("Loading buffer from {}".format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
