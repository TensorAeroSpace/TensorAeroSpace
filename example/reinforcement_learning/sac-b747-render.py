import numpy as np
import torch
from tqdm import tqdm

from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.b747 import ImprovedB747Env
from tensoraerospace.signals.standart import sinusoid_vertical_shift, unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

# Time and reference signal
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Discretization
dt = 0.1
# Time grid (indices and seconds)
tp = generate_time_period(tn=200, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

# Reference: sinusoid for theta (1 deg amplitude around 0, frequency 0.01 Hz)
reference_signals = np.reshape(
    sinusoid_vertical_shift(
        tp=np.asarray(tps),
        frequency=0.05,
        amplitude=np.deg2rad(1.0),
        vertical_shift=0.0,
    ),
    [1, -1],
)
# Step alternative:
# reference_signals = np.reshape(
#     unit_step(degree=1, tp=np.asarray(tps), time_step=5, output_rad=True),
#     [1, -1]
# )


# Build env: track theta and q, control elevator (stab)
initial_state = np.array([[0], [0], [0], [0]], dtype=np.float32)


env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference_signals,
    number_time_steps=number_time_steps,
    # tracking_states=["theta"],
    # state_space=["u", "w", "q", "theta"],
    # control_space=["stab"],
    # output_space=["u", "w", "q", "theta"],
    # reward_func=tracking_reward,
    # use_reward=True,
    initial_elevator_deg=0.0,  # задать стартовое положение руля
    use_initial_action_on_first_step=True,  # применить его на 1-м шаге
    dt=dt,
)

env.unwrapped.model.discretisation_time = dt

agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")

# Quick eval
state, info = env.reset()
done = False
ret = 0.0
actions = []
rewards = []
while not done:
    action = agent.select_action(state, evaluate=True)
    actions.append(action)
    state, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    env.render(mode="human")
    done = terminated or truncated
    ret += reward
