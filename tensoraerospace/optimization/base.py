from abc import ABC
from typing import Callable

import matplotlib.pyplot as plt
import optuna


class HyperParamOptimizationBase(ABC):
    """
        Класс интерфейс для поиск гиперпараметров
    """
    def __init__(self) -> None:
        """Инициализация
        """
        pass
    
    def run_optimization(self):
        """Запуск оптимизации
        """
        pass
    
    def get_best_param(self)->dict:
        """Получить лучшие найденные параметры

        Returns:
            dict: Лучшие параметры
        """
        pass

    def plot_parms(self, fig_size):
        """Построить график шагов оптимизации
        """
        pass
    

class HyperParamOptimizationOptuna(HyperParamOptimizationBase):
    """
        Поиск гиперпараметров модели
    """
    def __init__(self, direction:str) -> None:
        """Инициализация поиска гиперпараметров

        Args:
            direction (str): Направление поиска. Ex. minimize|maximaze
        """
        super().__init__()
        if direction not in ['minimize','maximaze' ]:
            raise ValueError("Выберите один из вариантов minimize или maximaze")
        self.study = optuna.create_study(direction=direction)
    
    def run_optimization(self, func:Callable, n_trials:int):
        """Запуск поиска гиперпараметров

        Args:
            func (void): Функция поиска параметров
            n_trials (int): Количество попыток для поиска
        """
        self.study.optimize(func, n_trials=n_trials)
    
    def get_best_param(self)->dict:
        """Получить лучшие гиперпараметров

        Returns:
            dict: Словарь с лучшими гиперпараметрами
        """
        return self.study.best_trial.params
    
    def plot_parms(self, figsize=(15, 5)):
        """Построить график поиска гиперпараметров

        Args:
            figsize (tuple, optional): Размер графика. По умолчанию (15, 5).
        """
        x = []
        x_labels = []
        for trial in self.study.trials:
            x.append(trial.value)
            x_labels.append("".join([f"{key}={trial.params[key]}\n" for key in trial.params.keys()]))
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(self.study.trials)), x)
        ax.set_xticks(range(len(self.study.trials)))
        ax.set_xticklabels(x_labels, rotation=90, multialignment="left")
        ax.set_title("График поиска гиперпараметров")
        ax.set_ylabel("Значении функции", fontsize=15)
        ax.set_xlabel("Итерации", fontsize=15)
