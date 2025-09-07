import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Модуль позиционного кодирования для трансформерных моделей.

    Добавляет позиционную информацию к входным эмбеддингам, используя синусоидальные функции.

    Args:
        d_model (int): Размерность модели (размер эмбеддингов).
        dropout (float): Вероятность dropout. По умолчанию 0.1.
        max_len (int): Максимальная длина последовательности. По умолчанию 5000.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Инициализация модуля позиционного кодирования.

        Args:
            d_model (int): Размерность модели.
            dropout (float): Вероятность dropout.
            max_len (int): Максимальная длина последовательности.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Применяет позиционное кодирование к входным данным.

        Args:
            x (torch.Tensor): Входной тензор размерности (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Тензор с добавленным позиционным кодированием.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDynamicsModel(nn.Module):
    """
    Трансформерная модель для моделирования динамики системы.

    Использует архитектуру трансформера для предсказания следующего состояния системы
    на основе текущего состояния и управляющего воздействия.

    Args:
        input_dim (int): Размерность входных данных (состояние + управление).
        output_dim (int): Размерность выходных данных (следующее состояние).
        d_model (int): Размерность модели трансформера. По умолчанию 64.
        nhead (int): Количество голов внимания. По умолчанию 4.
        num_encoder_layers (int): Количество слоев энкодера. По умолчанию 2.
        dim_feedforward (int): Размерность feed-forward сети. По умолчанию 256.
        dropout (float): Вероятность dropout. По умолчанию 0.1.
        seq_len (int): Длина последовательности. По умолчанию 1.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        seq_len=1,
    ):
        """
        Инициализация трансформерной модели динамики.

        Args:
            input_dim (int): Размерность входных данных.
            output_dim (int): Размерность выходных данных.
            d_model (int): Размерность модели трансформера.
            nhead (int): Количество голов внимания.
            num_encoder_layers (int): Количество слоев энкодера.
            dim_feedforward (int): Размерность feed-forward сети.
            dropout (float): Вероятность dropout.
            seq_len (int): Длина последовательности.
        """
        super(TransformerDynamicsModel, self).__init__()

        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, d_model)

        # self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Прямое распространение через трансформерную модель.

        Args:
            x (torch.Tensor): Входной тензор размерности (batch_size, input_dim).

        Returns:
            torch.Tensor: Предсказанное следующее состояние размерности (batch_size, output_dim).
        """
        # x: (batch_size, input_dim)

        x = x.unsqueeze(1)  # x: (batch_size, seq_len=1, input_dim)

        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.squeeze(1)  # (batch_size, d_model)
        x = self.fc_out(x)  # (batch_size, output_dim)
        return x
