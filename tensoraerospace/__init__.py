"""TensorAeroSpace - Reinforcement learning library for aerospace applications.

This package provides tools and environments for applying reinforcement learning methods
to aerospace system control tasks. It includes various models of aircraft, rockets,
and satellites, as well as control algorithms.

Main components:
    - Reinforcement learning environments (envs)
    - Reinforcement learning agents (agent)
    - Aerospace object models (aerospacemodel)
    - Control quality analysis tools (benchmark)
    - Utilities and helper functions (utils, signals)
"""

from gymnasium.envs.registration import register

register(
    id="LinearLongitudinalF16-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalF16",
)

register(
    id="LinearLongitudinalB747-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalB747",
)

register(
    id="LinearLongitudinalMissileModel-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalMissileModel",
)

register(
    id="LinearLongitudinalELVRocket-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalELVRocket",
)

register(
    id="LinearLongitudinalX15-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalX15",
)

register(
    id="LinearLongitudinalF4C-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalF4C",
)

register(
    id="LinearLongitudinalLAPAN-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalLAPAN",
)

register(
    id="LinearLongitudinalUltrastick-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalUltrastick",
)

register(
    id="LinearLongitudinalUAV-v0",
    entry_point="tensoraerospace.envs:LinearLongitudinalUAV",
)

register(
    id="GeoSat-v0",
    entry_point="tensoraerospace.envs:GeoSatEnv",
)

register(
    id="ComSat-v0",
    entry_point="tensoraerospace.envs:ComSatEnv",
)

# Improved environments registration
register(
    id="ImprovedB747-v0",
    entry_point="tensoraerospace.envs:ImprovedB747Env",
)

register(
    id="ImprovedComSat-v0",
    entry_point="tensoraerospace.envs:ImprovedComSatEnv",
)

register(
    id="ImprovedELV-v0",
    entry_point="tensoraerospace.envs:ImprovedELVEnv",
)

register(
    id="ImprovedLAPAN-v0",
    entry_point="tensoraerospace.envs:ImprovedLAPANEnv",
)

register(
    id="ImprovedMissile-v0",
    entry_point="tensoraerospace.envs:ImprovedMissileEnv",
)

register(
    id="ImprovedUltrastick-v0",
    entry_point="tensoraerospace.envs:ImprovedUltrastickEnv",
)

register(
    id="ImprovedX15-v0",
    entry_point="tensoraerospace.envs:ImprovedX15Env",
)

register(
    id="F4CPitchNormalized-v0",
    entry_point="tensoraerospace.envs:F4CPitchEnvNormalized",
)
