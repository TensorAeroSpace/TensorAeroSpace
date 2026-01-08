"""Transformer components used by MPC agents.

This module provides building blocks such as positional encoding and
transformer-based sequence models used in some MPC-related implementations.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Adds positional information to input embeddings using sinusoidal functions,
    as described in the original Transformer paper.

    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout probability. Defaults to ``0.1``.
        max_len (int): Maximum sequence length. Defaults to ``5000``.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize the positional encoding module.

        Args:
            d_model (int): Embedding dimension.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(seq_len, batch_size, d_model)``.

        Returns:
            torch.Tensor: Tensor with positional encoding applied.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDynamicsModel(nn.Module):
    """Transformer-based model for learning system dynamics.

    Predicts the next system state from a sequence of state+action inputs using
    a Transformer encoder.

    Args:
        input_dim (int): Input dimension (state + control).
        output_dim (int): Output dimension (next state).
        d_model (int): Transformer model dimension. Defaults to ``64``.
        nhead (int): Number of attention heads. Defaults to ``4``.
        num_encoder_layers (int): Number of encoder layers. Defaults to ``2``.
        dim_feedforward (int): Feed-forward layer dimension. Defaults to ``256``.
        dropout (float): Dropout probability. Defaults to ``0.1``.
        seq_len (int): Sequence length. Defaults to ``1``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 1,
    ) -> None:
        """Initialize the transformer dynamics model.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            dim_feedforward: Feed-forward dimension.
            dropout: Dropout probability.
            seq_len: Sequence length.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted next state of shape (batch_size, output_dim).
        """
        # x: (batch_size, input_dim)

        x = x.unsqueeze(1)  # x: (batch_size, seq_len=1, input_dim)

        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.squeeze(1)  # (batch_size, d_model)
        x = self.fc_out(x)  # (batch_size, output_dim)
        return x
