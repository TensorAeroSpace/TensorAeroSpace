from gym.envs.registration import register

register(
    id='LinearLongitudinalF16-v0',
    entry_point='tensorairspace.envs:LinearLongitudinalF16',
)

register(
    id='LinearLongitudinalB747-v0',
    entry_point='tensorairspace.envs:LinearLongitudinalB747',
)

register(
    id='LinearLongitudinalELVRocket-v0',
    entry_point='tensorairspace.envs:LinearLongitudinalELVRocket',
)