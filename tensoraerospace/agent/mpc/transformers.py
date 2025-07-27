import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDynamicsModel(nn.Module):
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
        # x: (batch_size, input_dim)

        x = x.unsqueeze(1)  # x: (batch_size, seq_len=1, input_dim)

        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.squeeze(1)  # (batch_size, d_model)
        x = self.fc_out(x)  # (batch_size, output_dim)
        return x
