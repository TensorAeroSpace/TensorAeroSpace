from gymnasium.envs.registration import register

register(
    id='LinearLongitudinalF16-v0',
    entry_point='tensoraerospace.envs:LinearLongitudinalF16',
)

register(
    id='LinearLongitudinalB747-v0',
    entry_point='tensoraerospace.envs:LinearLongitudinalB747',
)

# register(
#     id='LinearLongitudinalMissileModel-v0',
#     entry_point='tensoraerospace.envs:LinearLongitudinalMissileModel',
# )

# register(
#     id='LinearLongitudinalELVRocket-v0',
#     entry_point='tensoraerospace.envs:LinearLongitudinalELVRocket',
# )

# register(
#     id='LinearLongitudinalX15-v0',
#     entry_point='tensoraerospace.envs:LinearLongitudinalX15',
# )

# register(
#     id='LinearLongitudinalF4C-v0',
#     entry_point='tensoraerospace.envs:LinearLongitudinalF4C',
# )

register(
    id='LinearLongitudinalUAV-v0',
    entry_point='tensoraerospace.envs:LinearLongitudinalUAV',
)

# register(
#     id='GeoSat-v0',
#     entry_point='tensoraerospace.envs:GeoSatEnv',
# )

# register(
#     id='ComSat-v0',
#     entry_point='tensoraerospace.envs:ComSatEnv',
# )