"""Smoke test for l4 package skeleton."""
from __future__ import annotations


def test_l4_package_importable() -> None:
    import tensoraerospace.agent.uftc.l4 as l4
    assert hasattr(l4, "__all__")
    assert "DSACConfig" in l4.__all__


def test_dsac_config_defaults() -> None:
    from tensoraerospace.agent.uftc.l4 import DSACConfig
    cfg = DSACConfig(n_state=4, n_ref_dim=4, n_action=4)
    assert cfg.cvar_alpha == 0.2
    assert cfg.gamma == 0.99
    assert cfg.eval_mode is True
