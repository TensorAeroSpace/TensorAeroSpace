import torch
import torch.nn as nn


class NARX(nn.Module):
    """Модель нейронной сети NARX (Nonlinear AutoRegressive with eXogenous inputs),
    основанная на полносвязных слоях для предсказания временных рядов.

    Args:
        input_size (int): Размер входного вектора.
        hidden_size (int): Размер скрытого слоя.
        output_size (int): Размер выходного вектора.

    Attributes:
        hidden_size (int): Размер скрытого слоя.
        input_layer (nn.Linear): Полносвязный слой, принимающий на вход комбинацию
                                 предыдущего выхода и текущего входа.
        output_layer (nn.Linear): Полносвязный слой для получения предсказания выходного значения.
        criterion (nn.MSELoss): Функция потерь среднеквадратичной ошибки.
        optimizer (torch.optim.Adam): Оптимизатор Adam.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(NARX, self).__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + output_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)


    def forward(self, input_tensor, last_output):
        """Выполняет один шаг прямого распространения сигнала.

        Args:
            input_tensor (Tensor): Тензор входных данных.
            last_output (Tensor): Тензор последнего выходного значения модели.

        Returns:
            Tensor: Тензор выходных данных модели.
        """
        combined = torch.cat((input_tensor, last_output), 0)
        hidden = torch.tanh(self.input_layer(combined))
        output = self.output_layer(hidden)
        return output
    
    def train(self, predcit_tensor, target_tensor):
        """Обучает модель, минимизируя функцию потерь между предсказанным и целевым тензором.

        Args:
            predict_tensor (Tensor): Тензор предсказаний модели.
            target_tensor (Tensor): Тензор целевых значений.

        Returns:
            float: Значение функции потерь после одного шага обучения.

        """
        loss = self.criterion(predcit_tensor, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
