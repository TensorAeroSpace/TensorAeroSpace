from typing import Callable

from ray import tune

from .base import HyperParamOptimizationBase


class HyperParamOptimizationRay(HyperParamOptimizationBase):
    """
        Поиск гиперпараметров модели
    """
    def __init__(self, direction:str) -> None:
        """Инициализация поиска гиперпараметров

        Args:
            direction (str): Направление поиска. Ex. minimize|maximaze
        """
        super().__init__()
    
    def run_optimization(self, func:Callable, param_space, tune_config=tune.TuneConfig(num_samples=5), **kwargs):
        """Запуск поиска гиперпараметров

        Args:
            func (Callable): Функция поиска параметров
            param_space (_type_): Переменные для поиска
            tune_config (_type_, optional): Параметры оптимизации. Defaults to tune.TuneConfig(num_samples=5).
        """
        self.tuner = tune.Tuner(func, param_space=param_space, tune_config=tune_config, **kwargs)
        self.results = self.tuner.fit()

    def get_best_param(self)->dict:
        """Получить лучшие гиперпараметров

        Returns:
            dict: Словарь с лучшими гиперпараметрами
        """
        return self.study.best_trial.params
    
    def plot_parms(self):
        """Построить график поиска гиперпараметров (WIP)

        Raises:
            NotImplementedError:  (WIP)
        """
        raise NotImplementedError()
