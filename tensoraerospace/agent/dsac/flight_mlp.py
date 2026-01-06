"""MLP builder copied from dsac-flight.

This is intentionally kept close to the original implementation to make
DSAC behaviour match the reference repo.
"""

from __future__ import annotations

from typing import Any, Optional

from torch import nn


def make_mlp(
    *,
    num_in: int,
    num_out: int,
    n_hidden_layers: int,
    n_hidden_units: int,
    final_activation: Optional[Any] = None,
) -> nn.Sequential:
    """Build a MLP as torch.nn.Sequential (dsac-flight style)."""
    layers: list[nn.Module] = []

    layers.extend(
        [
            nn.Linear(num_in, n_hidden_units),
            nn.ReLU(),
        ]
    )

    for _ in range(int(n_hidden_layers)):
        layers.extend(
            [
                nn.Linear(n_hidden_units, n_hidden_units),
                nn.LayerNorm(n_hidden_units),
                nn.ReLU(),
            ]
        )

    layers.append(nn.Linear(n_hidden_units, num_out))
    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)
