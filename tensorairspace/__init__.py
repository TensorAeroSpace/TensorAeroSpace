from gym.envs.registration import register

register(
    id='LinearLongitudinalF16-v0',
    entry_point='tensorairspace.envs.f16:LinearLongitudinalF16',
)

register(
    id='LinearLongitudinalB747-v0',
    entry_point='tensorairspace.envs.b747:LinearLongitudinalB747',
)

