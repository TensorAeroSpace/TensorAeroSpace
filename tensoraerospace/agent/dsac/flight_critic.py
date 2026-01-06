"""Distributional critic copied from dsac-flight (IQN / ZNet)."""

from __future__ import annotations

import torch
from torch import nn

from .flight_mlp import make_mlp


class ZNet(nn.Module):
    """Wrapper around IQN that outputs Z(s, a; taus) with shape (B, N)."""

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        n_hidden_layers: int,
        n_hidden_units: int,
        n_cos: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.iqn = IQN(
            n_inputs=int(n_states + n_actions),
            n_outputs=1,
            n_hidden_layers=int(n_hidden_layers),
            n_hidden_units=int(n_hidden_units),
            embedding_size=int(n_cos),
            device=device,
        ).to(self.device)

    @staticmethod
    def generate_taus(
        *, batch_size: int, n_taus: int, device: torch.device
    ) -> torch.Tensor:
        """Uniform taus in (0,1), shape (B, N)."""
        with torch.no_grad():
            return torch.rand(int(batch_size), int(n_taus), device=device)

    def forward(
        self, s: torch.Tensor, a: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([s, a], dim=1)
        taus = taus.unsqueeze(-1)  # (B, N, 1)
        z = self.iqn(x, taus)  # (B, N, 1)
        return z.squeeze(2)  # (B, N)


class IQN(nn.Module):
    """Implicit Quantile Network (dsac-flight style)."""

    def __init__(
        self,
        *,
        n_inputs: int,
        n_outputs: int,
        embedding_size: int,
        n_hidden_layers: int,
        n_hidden_units: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

        self.S = int(n_inputs)
        self.A = int(n_outputs)
        self.C = int(embedding_size)
        self.H = int(n_hidden_units)

        self.input_layer = nn.Sequential(
            nn.Linear(self.S, self.H),
            nn.LayerNorm(self.H),
            nn.ReLU(),
        )

        self.const_pi_vec = (
            torch.arange(start=0, end=self.C, device=self.device) * torch.pi
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.C, self.H),
            nn.LayerNorm(self.H),
            nn.Sigmoid(),
        )

        self.hidden_layers = make_mlp(
            num_in=self.H,
            num_out=self.A,
            n_hidden_layers=int(n_hidden_layers),
            n_hidden_units=self.H,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        # x: (B, S), taus: (B, N, 1)
        B = x.shape[0]
        N = taus.shape[1]
        C = self.C
        H = self.H

        x = self.input_layer(x)  # (B, H)

        cos = torch.cos(taus * self.const_pi_vec)  # (B, N, C)
        cos = cos.view(B * N, C)  # (B*N, C)
        cos_out = self.embedding_layer(cos)  # (B*N, H)
        cos_out = cos_out.view(B, N, H)  # (B, N, H)

        x_reshaped = x.unsqueeze(1)  # (B, 1, H)
        h = torch.mul(x_reshaped, cos_out)  # (B, N, H)

        h = h.view(B * N, H)  # (B*N, H)
        out = self.hidden_layers(h)  # (B*N, A)
        out = out.view(B, N, self.A)  # (B, N, A)
        return out
