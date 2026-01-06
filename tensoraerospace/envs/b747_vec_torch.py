"""Torch vectorized Boeing 747 longitudinal environment (GPU-friendly).

This module provides a batched version of the ImprovedB747Env for fast PPO training.
It simulates N independent B747 linear longitudinal models in parallel on a chosen
torch device (CPU or CUDA).

Key goals:
- Keep all step computations in torch (optionally on GPU)
- Vectorize over num_envs (e.g. 64)
- Randomize step amplitude and step time per episode to prevent anticipation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces
from scipy.signal import cont2discrete


@dataclass
class SignalRandomization:
    """Parameters for randomized reference signals.

    Supports multiple signal types: step, sine, ramp, and mixed.

    Attributes:
        signal_type: Type of signal to generate. Options:
            - "step": Unit step at random time with random amplitude
            - "sine": Sinusoidal signal with random amplitude and frequency
            - "ramp": Linear ramp with random slope
            - "mixed": Randomly choose between step/sine/ramp each episode
        amplitude_deg_range: Min/max amplitude in degrees
        min_abs_amplitude_deg: Minimum absolute amplitude to avoid trivial signals
        step_time_sec_range: For step signals, when the step occurs (sec)
        frequency_hz_range: For sine signals, frequency range (Hz)
        p_step: Probability of step signal in "mixed" mode
        p_sine: Probability of sine signal in "mixed" mode (ramp = 1 - p_step - p_sine)
        ref_change_threshold_rad: Threshold for detecting reference changes
        min_step_amp_rad: Minimum step amplitude for overshoot metrics
    """

    signal_type: str = "step"
    amplitude_deg_range: Tuple[float, float] = (1.0, 10.0)
    min_abs_amplitude_deg: float = 1.0
    step_time_sec_range: Tuple[float, float] = (1.0, 10.0)
    frequency_hz_range: Tuple[float, float] = (0.02, 0.1)
    p_step: float = 0.5
    p_sine: float = 0.3
    ref_change_threshold_rad: float = float(np.deg2rad(0.1))
    min_step_amp_rad: float = float(np.deg2rad(0.5))


def _make_signal_randomization(params: dict) -> SignalRandomization:
    """Create SignalRandomization from dict, ignoring unknown fields for backward compatibility."""
    known_fields = {f.name for f in SignalRandomization.__dataclass_fields__.values()}
    filtered = {k: v for k, v in params.items() if k in known_fields}
    return SignalRandomization(**filtered)


# Backward compatibility: StepRandomization was the old name
StepRandomization = SignalRandomization


class ImprovedB747VecEnvTorch:
    """Vectorized torch implementation of ImprovedB747Env.

    Notes:
        - Observations are already normalized to [-1, 1], so PPO can typically
          run with normalize_obs=False for speed.
        - This env returns torch tensors on `device`.
        - API is Gymnasium-like but batched:
            reset() -> (obs[N, obs_dim], info)
            step(action[N, act_dim]) -> (obs, reward[N], terminated[N], truncated[N], info)
    """

    def __init__(
        self,
        *,
        num_envs: int = 64,
        dt: float = 0.1,
        tn: float = 20.0,
        initial_state: Optional[np.ndarray] = None,
        device: Optional[torch.device | str] = None,
        seed: Optional[int] = None,
        auto_reset: bool = True,
        include_reference_in_obs: bool = False,
        step_randomization: Optional[StepRandomization | dict[str, Any]] = None,
        reward_mode: str = "step_response",
        survival_bonus: float = 0.0,
        completion_bonus: float = 0.0,
        early_termination_penalty: float = 0.0,
        early_termination_penalty_per_step: float = 0.0,
    ) -> None:
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        # Reward mode: "tracking" (universal) or "step_response" (with step-specific penalties)
        if reward_mode not in ("tracking", "step_response"):
            raise ValueError(
                f"reward_mode must be 'tracking' or 'step_response', got {reward_mode!r}"
            )
        self.reward_mode = str(reward_mode)
        self.survival_bonus = float(survival_bonus)
        self.completion_bonus = float(completion_bonus)
        self.early_termination_penalty = float(early_termination_penalty)
        self.early_termination_penalty_per_step = float(
            early_termination_penalty_per_step
        )
        if self.survival_bonus < 0:
            raise ValueError("survival_bonus must be >= 0")
        if self.completion_bonus < 0:
            raise ValueError("completion_bonus must be >= 0")
        if self.early_termination_penalty < 0:
            raise ValueError("early_termination_penalty must be >= 0")
        if self.early_termination_penalty_per_step < 0:
            raise ValueError("early_termination_penalty_per_step must be >= 0")

        self.dt = float(dt)
        self.tn = float(tn)
        self.number_time_steps = int(np.round(self.tn / self.dt)) + 1

        self.device = (
            torch.device("cuda")
            if (device is None and torch.cuda.is_available())
            else torch.device("cpu" if device is None else device)
        )
        self.auto_reset = bool(auto_reset)
        self.include_reference_in_obs = bool(include_reference_in_obs)

        # RNG for reference sampling (torch generator on selected device)
        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

        # Initial state [u, w, q, theta] in SI units (rad for angles)
        if initial_state is None:
            initial_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x0 = np.array(initial_state, dtype=np.float32).reshape(-1)
        if x0.shape[0] != 4:
            raise ValueError("initial_state must have 4 elements: [u, w, q, theta]")
        self._x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)

        # Physical constraints and normalization parameters (same as ImprovedB747Env)
        self.max_pitch_rad = float(np.deg2rad(20.0))
        self.max_pitch_rate_rad_s = float(np.deg2rad(5.0))
        self.max_stabilizer_angle_deg = 25.0
        self._max_elev_rad = float(np.deg2rad(self.max_stabilizer_angle_deg))

        # Spaces (single-env shapes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = 6 if self.include_reference_in_obs else 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Discrete-time system matrices (computed once, then moved to torch)
        A = np.array(
            [
                [-0.0069, 0.0139, 0.0, -9.8100],
                [-0.0905, -0.3149, 235.8928, 0.0],
                [0.0004, -0.0034, -0.4282, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        B = np.array([[-0.0001], [-5.5079], [-1.1569], [0.0]], dtype=np.float64)
        C = np.eye(4, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), self.dt)
        self.Ad = torch.as_tensor(Ad.astype(np.float32), device=self.device)
        self.Bd = torch.as_tensor(Bd.astype(np.float32), device=self.device)

        # Input constraints (mirror LongitudinalB747 behavior)
        self.input_magnitude_limit_rad = float(np.deg2rad(25.0))
        self.input_rate_limit = 60.0  # model uses it in same units as u (legacy)
        self._rate_step = float(self.input_rate_limit * self.dt)

        # Reward parameters (reuse tuned defaults from ImprovedB747Env)
        self.reward_scale = 0.1
        self.w_pitch = 5.0
        self.w_q = 0.2
        self.w_action = 0.003
        self.w_smooth = 0.01
        self.w_jerk = 0.001
        self.k_progress = 0.05

        # Strict step-response targets
        self.overshoot_limit_ratio = 0.05
        self.ref_theta_dot_clip_rad_s = float(self.max_pitch_rate_rad_s)
        self.settle_band_ratio = 0.01
        self.settle_band_min_rad = float(np.deg2rad(0.05))
        self.q_settle_rad_s = float(np.deg2rad(0.25))
        self.settle_steps_required = int(max(1, np.ceil(1.0 / self.dt)))
        self.settle_time_target_s = 1.5
        self.w_abs = 0.6
        self.w_time = 0.6
        self.w_osc = 1.0
        self.w_overshoot = 300.0
        self.w_settle_bonus = 4.0

        if step_randomization is None:
            self.step_rand = SignalRandomization()
        elif isinstance(step_randomization, dict):
            # Use helper to filter unknown fields (backward compatibility with old checkpoints)
            self.step_rand = _make_signal_randomization(step_randomization)
        else:
            self.step_rand = step_randomization

        # Time vector for reference generation
        self.tps = torch.linspace(
            0.0,
            self.tn,
            steps=self.number_time_steps,
            device=self.device,
            dtype=torch.float32,
        )

        # Internal state tensors (allocated in reset)
        self._alloc()
        # Store init args for serialization (PPO checkpoints)
        self._init_args: dict[str, Any] = {
            "num_envs": int(self.num_envs),
            "dt": float(self.dt),
            "tn": float(self.tn),
            "initial_state": np.array(x0, dtype=np.float32),
            "device": str(self.device),
            "seed": int(seed) if seed is not None else None,
            "auto_reset": bool(self.auto_reset),
            "include_reference_in_obs": bool(self.include_reference_in_obs),
            "step_randomization": {
                "signal_type": str(getattr(self.step_rand, "signal_type", "step")),
                "amplitude_deg_range": tuple(self.step_rand.amplitude_deg_range),
                "min_abs_amplitude_deg": float(
                    getattr(self.step_rand, "min_abs_amplitude_deg", 0.0)
                ),
                "step_time_sec_range": tuple(self.step_rand.step_time_sec_range),
                "frequency_hz_range": tuple(
                    getattr(self.step_rand, "frequency_hz_range", (0.02, 0.1))
                ),
                "p_step": float(getattr(self.step_rand, "p_step", 0.5)),
                "p_sine": float(getattr(self.step_rand, "p_sine", 0.3)),
                "ref_change_threshold_rad": float(
                    self.step_rand.ref_change_threshold_rad
                ),
                "min_step_amp_rad": float(self.step_rand.min_step_amp_rad),
            },
            "reward_mode": str(self.reward_mode),
            "survival_bonus": float(self.survival_bonus),
            "completion_bonus": float(self.completion_bonus),
            "early_termination_penalty": float(self.early_termination_penalty),
            "early_termination_penalty_per_step": float(
                self.early_termination_penalty_per_step
            ),
        }
        self.reset()

    @property
    def unwrapped(self) -> "ImprovedB747VecEnvTorch":
        # Gymnasium compatibility for serialization utilities
        return self

    def get_init_args(self) -> dict[str, Any]:
        """Return init kwargs for serialize_env()."""
        return dict(self._init_args)

    @property
    def _idx_q(self) -> int:
        return 2

    @property
    def _idx_theta(self) -> int:
        return 3

    def _alloc(self) -> None:
        n = self.num_envs
        self.state = self._x0.repeat(n, 1).clone()
        self.step_count = torch.zeros((n,), device=self.device, dtype=torch.int64)

        # Action history (normalized)
        self.prev_action = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self.pre_prev_action = torch.zeros(
            (n,), device=self.device, dtype=torch.float32
        )
        # Previous applied input in rad (for rate limiting)
        self.prev_u_rad = torch.zeros((n, 1), device=self.device, dtype=torch.float32)

        # Reference signals per env: (n, T)
        self.reference_signal = torch.zeros(
            (n, self.number_time_steps), device=self.device, dtype=torch.float32
        )

        # Shaping state
        self._prev_e_theta = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self._prev_e_q_rel = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self._prev_target_theta = torch.zeros(
            (n,), device=self.device, dtype=torch.float32
        )
        self._seg_start_step = torch.zeros((n,), device=self.device, dtype=torch.int64)
        self._seg_amp = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self._seg_sign = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self._seg_max_err_dir = torch.zeros(
            (n,), device=self.device, dtype=torch.float32
        )
        self._settle_count = torch.zeros((n,), device=self.device, dtype=torch.int64)
        self._is_settled = torch.zeros((n,), device=self.device, dtype=torch.bool)
        self._settle_time_s = torch.full(
            (n,), -1.0, device=self.device, dtype=torch.float32
        )
        self._prev_error_sign = torch.zeros((n,), device=self.device, dtype=torch.int64)
        self._sign_changes = torch.zeros((n,), device=self.device, dtype=torch.int64)

    def _sample_amplitude(self, n: int) -> torch.Tensor:
        """Sample amplitude in degrees with deadzone handling."""
        amp_lo, amp_hi = self.step_rand.amplitude_deg_range
        min_abs = float(getattr(self.step_rand, "min_abs_amplitude_deg", 0.0) or 0.0)

        if (amp_lo < 0.0) and (amp_hi > 0.0) and (min_abs > 0.0):
            # Symmetric sampling excluding a deadzone around 0.
            max_abs = float(max(abs(amp_lo), abs(amp_hi)))
            mag_lo = float(min(min_abs, max_abs))
            mag = mag_lo + (max_abs - mag_lo) * torch.rand(
                (n,), generator=self._gen, device=self.device
            )
            sign = torch.where(
                torch.rand((n,), generator=self._gen, device=self.device) < 0.5,
                -torch.ones((n,), device=self.device),
                torch.ones((n,), device=self.device),
            )
            amp_deg = sign * mag
        else:
            amp_deg = amp_lo + (amp_hi - amp_lo) * torch.rand(
                (n,), generator=self._gen, device=self.device
            )
            if min_abs > 0.0:
                s = torch.sign(amp_deg)
                s = torch.where(s == 0.0, torch.ones_like(s), s)
                amp_deg = torch.where(
                    torch.abs(amp_deg) < min_abs, s * min_abs, amp_deg
                )
        return amp_deg

    def _build_step_reference(self, n: int, amp_rad: torch.Tensor) -> torch.Tensor:
        """Build step reference signals."""
        st_lo, st_hi = self.step_rand.step_time_sec_range
        step_time = st_lo + (st_hi - st_lo) * torch.rand(
            (n,), generator=self._gen, device=self.device
        )
        step_time = torch.clamp(
            step_time, float(self.tps[0].item()), float(self.tps[-1].item())
        )
        t = self.tps[None, :]  # (1, T)
        st = step_time[:, None]  # (n, 1)
        return amp_rad[:, None] * (t >= st).to(torch.float32)

    def _build_sine_reference(self, n: int, amp_rad: torch.Tensor) -> torch.Tensor:
        """Build sinusoidal reference signals."""
        freq_lo, freq_hi = getattr(self.step_rand, "frequency_hz_range", (0.02, 0.1))
        freq = freq_lo + (freq_hi - freq_lo) * torch.rand(
            (n,), generator=self._gen, device=self.device
        )
        # Random phase for diversity
        phase = 2.0 * np.pi * torch.rand((n,), generator=self._gen, device=self.device)
        t = self.tps[None, :]  # (1, T)
        f = freq[:, None]  # (n, 1)
        p = phase[:, None]  # (n, 1)
        return amp_rad[:, None] * torch.sin(2.0 * np.pi * f * t + p)

    def _build_ramp_reference(self, n: int, amp_rad: torch.Tensor) -> torch.Tensor:
        """Build ramp reference signals (linear increase to target)."""
        # Ramp starts at random time and reaches target by end
        ramp_start_lo, ramp_start_hi = 0.0, float(self.tn * 0.3)
        ramp_start = ramp_start_lo + (ramp_start_hi - ramp_start_lo) * torch.rand(
            (n,), generator=self._gen, device=self.device
        )
        ramp_duration = float(self.tn) - ramp_start  # Time to reach target
        t = self.tps[None, :]  # (1, T)
        rs = ramp_start[:, None]  # (n, 1)
        rd = ramp_duration[:, None]  # (n, 1)
        # Linear ramp: 0 before start, then linear to amp_rad
        progress = torch.clamp((t - rs) / (rd + 1e-6), 0.0, 1.0)
        return amp_rad[:, None] * progress

    def _sample_reference_for_indices(self, idx: torch.Tensor) -> None:
        """Sample reference signals for envs in idx based on signal_type."""
        if idx.numel() == 0:
            return

        n = int(idx.numel())
        signal_type = str(getattr(self.step_rand, "signal_type", "step"))

        # Sample amplitude (common for all signal types)
        amp_deg = self._sample_amplitude(n)
        amp_rad = torch.deg2rad(amp_deg)

        if signal_type == "mixed":
            # Randomly assign signal type per environment
            p_step = float(getattr(self.step_rand, "p_step", 0.5))
            p_sine = float(getattr(self.step_rand, "p_sine", 0.3))
            rand_vals = torch.rand((n,), generator=self._gen, device=self.device)

            # Initialize reference tensor
            ref = torch.zeros(
                (n, self.number_time_steps), device=self.device, dtype=torch.float32
            )

            # Step signals
            step_mask = rand_vals < p_step
            if torch.any(step_mask):
                n_step = int(step_mask.sum().item())
                step_ref = self._build_step_reference(n_step, amp_rad[step_mask])
                ref[step_mask] = step_ref

            # Sine signals
            sine_mask = (rand_vals >= p_step) & (rand_vals < p_step + p_sine)
            if torch.any(sine_mask):
                n_sine = int(sine_mask.sum().item())
                sine_ref = self._build_sine_reference(n_sine, amp_rad[sine_mask])
                ref[sine_mask] = sine_ref

            # Ramp signals (remaining)
            ramp_mask = rand_vals >= (p_step + p_sine)
            if torch.any(ramp_mask):
                n_ramp = int(ramp_mask.sum().item())
                ramp_ref = self._build_ramp_reference(n_ramp, amp_rad[ramp_mask])
                ref[ramp_mask] = ramp_ref

        elif signal_type == "sine":
            ref = self._build_sine_reference(n, amp_rad)

        elif signal_type == "ramp":
            ref = self._build_ramp_reference(n, amp_rad)

        else:  # "step" (default)
            ref = self._build_step_reference(n, amp_rad)

        self.reference_signal[idx] = ref

        # Reset segment tracking for these envs
        self._prev_target_theta[idx] = self.reference_signal[idx, 0]
        self._seg_start_step[idx] = 0
        self._seg_amp[idx] = 0.0
        self._seg_sign[idx] = 0.0
        self._seg_max_err_dir[idx] = 0.0
        self._settle_count[idx] = 0
        self._is_settled[idx] = False
        self._settle_time_s[idx] = -1.0
        self._prev_error_sign[idx] = 0
        self._sign_changes[idx] = 0
        self._prev_e_theta[idx] = 0.0
        self._prev_e_q_rel[idx] = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        if seed is not None:
            self._gen.manual_seed(int(seed))

        n = self.num_envs
        self.state = self._x0.repeat(n, 1).clone()
        self.step_count.zero_()
        self.prev_action.zero_()
        self.pre_prev_action.zero_()
        self.prev_u_rad.zero_()

        # Sample new reference for all envs
        self._sample_reference_for_indices(
            torch.arange(n, device=self.device, dtype=torch.int64)
        )

        return self._get_obs(), {}

    def _get_obs(self) -> torch.Tensor:
        theta = self.state[:, self._idx_theta]
        q = self.state[:, self._idx_q]
        idx = torch.clamp(self.step_count, 0, self.number_time_steps - 1)
        idx_prev = torch.clamp(self.step_count - 1, 0, self.number_time_steps - 1)
        target = self.reference_signal[
            torch.arange(self.num_envs, device=self.device), idx
        ]
        target_prev = self.reference_signal[
            torch.arange(self.num_envs, device=self.device), idx_prev
        ]

        pitch_error = target - theta
        norm_pitch_error = torch.clamp(pitch_error / self.max_pitch_rad, -1.0, 1.0)
        norm_q = torch.clamp(q / self.max_pitch_rate_rad_s, -1.0, 1.0)
        norm_theta = torch.clamp(theta / self.max_pitch_rad, -1.0, 1.0)
        norm_prev_action = torch.clamp(self.prev_action, -1.0, 1.0)
        if not self.include_reference_in_obs:
            return torch.stack(
                [norm_pitch_error, norm_q, norm_theta, norm_prev_action], dim=-1
            )

        ref_theta_dot = (target - target_prev) / float(self.dt)
        ref_theta_dot = torch.clamp(
            ref_theta_dot,
            -float(self.ref_theta_dot_clip_rad_s),
            float(self.ref_theta_dot_clip_rad_s),
        )
        norm_target_theta = torch.clamp(target / self.max_pitch_rad, -1.0, 1.0)
        norm_ref_theta_dot = torch.clamp(
            ref_theta_dot / self.max_pitch_rate_rad_s, -1.0, 1.0
        )
        return torch.stack(
            [
                norm_pitch_error,
                norm_q,
                norm_theta,
                norm_prev_action,
                norm_target_theta,
                norm_ref_theta_dot,
            ],
            dim=-1,
        )

    def _reset_done(self, done: torch.Tensor) -> None:
        """Reset only environments where done=True."""
        idx = torch.where(done)[0]
        if idx.numel() == 0:
            return
        self.state[idx] = self._x0
        self.step_count[idx] = 0
        self.prev_action[idx] = 0.0
        self.pre_prev_action[idx] = 0.0
        self.prev_u_rad[idx] = 0.0
        self._sample_reference_for_indices(idx)

    def step(self, action: torch.Tensor):
        """Step all environments in parallel.

        Args:
            action: Tensor of shape (num_envs, 1) or (num_envs,) in [-1, 1]
        """
        # Ensure action shape (N, 1)
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.to(self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        action = torch.clamp(action, -1.0, 1.0)

        # Convert normalized action to radians and apply rate + magnitude limits
        u_deg = action.squeeze(-1) * float(self.max_stabilizer_angle_deg)
        u_rad_cmd = torch.deg2rad(u_deg).unsqueeze(-1)  # (N, 1)

        # Rate limit around previous applied u
        lo = self.prev_u_rad - float(self._rate_step)
        hi = self.prev_u_rad + float(self._rate_step)
        u_rad = torch.clamp(u_rad_cmd, lo, hi)
        # Magnitude limit
        u_rad = torch.clamp(
            u_rad,
            -float(self.input_magnitude_limit_rad),
            float(self.input_magnitude_limit_rad),
        )

        # Discrete dynamics: x_{t+1} = A_d x_t + B_d u_t
        x = self.state  # (N, 4)
        x_next = x @ self.Ad.T + u_rad @ self.Bd.T  # (N, 4)
        self.state = x_next
        self.prev_u_rad = u_rad

        # Step counter update
        self.step_count = self.step_count + 1

        # References
        idx = torch.clamp(self.step_count, 0, self.number_time_steps - 1)
        idx_prev = torch.clamp(self.step_count - 1, 0, self.number_time_steps - 1)
        ar = torch.arange(self.num_envs, device=self.device)
        target = self.reference_signal[ar, idx]
        target_prev = self.reference_signal[ar, idx_prev]
        ref_theta_dot = (target - target_prev) / float(self.dt)
        ref_theta_dot = torch.clamp(
            ref_theta_dot,
            -float(self.ref_theta_dot_clip_rad_s),
            float(self.ref_theta_dot_clip_rad_s),
        )

        theta = self.state[:, self._idx_theta]
        q = self.state[:, self._idx_q]

        # Base cost
        e_theta = (theta - target) / float(self.max_pitch_rad)
        e_q_rel = (q - ref_theta_dot) / float(self.max_pitch_rate_rad_s)

        u_norm = (u_rad.squeeze(-1) / float(self._max_elev_rad)).clamp(-1.0, 1.0)
        du = u_norm - self.prev_action
        ddu = u_norm - 2.0 * self.prev_action + self.pre_prev_action
        cost_base = (
            self.w_pitch * (e_theta**2)
            + self.w_q * (e_q_rel**2)
            + self.w_action * (u_norm**2)
            + self.w_smooth * (du**2)
            + self.w_jerk * (ddu**2)
        )

        # Progress shaping
        progress = self.k_progress * (
            (self._prev_e_theta**2 + self._prev_e_q_rel**2) - (e_theta**2 + e_q_rel**2)
        )
        self._prev_e_theta = e_theta.detach()
        self._prev_e_q_rel = e_q_rel.detach()

        # Detect new reference segment (step start)
        ref_delta = target - self._prev_target_theta
        new_seg = torch.abs(ref_delta) > float(self.step_rand.ref_change_threshold_rad)
        if torch.any(new_seg):
            self._seg_start_step = torch.where(
                new_seg, self.step_count, self._seg_start_step
            )
            self._seg_amp = torch.where(new_seg, ref_delta, self._seg_amp)
            self._seg_sign = torch.where(new_seg, torch.sign(ref_delta), self._seg_sign)
            self._seg_max_err_dir = torch.where(
                new_seg, torch.zeros_like(self._seg_max_err_dir), self._seg_max_err_dir
            )
            self._settle_count = torch.where(
                new_seg, torch.zeros_like(self._settle_count), self._settle_count
            )
            self._is_settled = torch.where(
                new_seg, torch.zeros_like(self._is_settled), self._is_settled
            )
            self._settle_time_s = torch.where(
                new_seg, torch.full_like(self._settle_time_s, -1.0), self._settle_time_s
            )
            self._prev_error_sign = torch.where(
                new_seg, torch.zeros_like(self._prev_error_sign), self._prev_error_sign
            )
            self._sign_changes = torch.where(
                new_seg, torch.zeros_like(self._sign_changes), self._sign_changes
            )
        self._prev_target_theta = target.detach()

        amp_abs = torch.abs(self._seg_amp)
        band_rad = torch.where(
            amp_abs >= float(self.step_rand.min_step_amp_rad),
            torch.maximum(
                amp_abs * float(self.settle_band_ratio),
                torch.full_like(amp_abs, float(self.settle_band_min_rad)),
            ),
            torch.full_like(amp_abs, float(self.settle_band_min_rad)),
        )

        err_theta = theta - target
        inside_band = (torch.abs(err_theta) <= band_rad) & (
            torch.abs(q) <= float(self.q_settle_rad_s)
        )
        self._settle_count = torch.where(
            inside_band, self._settle_count + 1, torch.zeros_like(self._settle_count)
        )

        just_settled = (~self._is_settled) & (
            self._settle_count >= int(self.settle_steps_required)
        )
        if torch.any(just_settled):
            self._is_settled = self._is_settled | just_settled
            settle_time = (self.step_count - self._seg_start_step).to(
                torch.float32
            ) * float(self.dt)
            self._settle_time_s = torch.where(
                just_settled, settle_time, self._settle_time_s
            )

        # Overshoot ratio
        overshoot_ratio = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        overshoot_excess = torch.zeros_like(overshoot_ratio)
        active_step = (amp_abs >= float(self.step_rand.min_step_amp_rad)) & (
            self._seg_sign != 0.0
        )
        if torch.any(active_step):
            err_dir = err_theta * self._seg_sign
            self._seg_max_err_dir = torch.where(
                active_step,
                torch.maximum(self._seg_max_err_dir, err_dir),
                self._seg_max_err_dir,
            )
            overshoot = torch.clamp(self._seg_max_err_dir, min=0.0)
            overshoot_ratio = torch.where(
                active_step, overshoot / (amp_abs + 1e-9), overshoot_ratio
            )
            overshoot_excess = torch.clamp(
                overshoot_ratio - float(self.overshoot_limit_ratio), min=0.0
            )

        # Oscillations: penalize repeat crossings (more than 1 sign change) outside band
        osc_event = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        outside_band = torch.abs(err_theta) > band_rad
        sgn = torch.sign(err_theta).to(torch.int64)
        crossing = (
            outside_band
            & (sgn != 0)
            & (self._prev_error_sign != 0)
            & (sgn != self._prev_error_sign)
        )
        new_sign_changes = self._sign_changes + crossing.to(torch.int64)
        osc_event = torch.where(
            crossing & (new_sign_changes > 1), torch.ones_like(osc_event), osc_event
        )
        self._sign_changes = new_sign_changes
        self._prev_error_sign = torch.where(
            outside_band & (sgn != 0), sgn, self._prev_error_sign
        )

        # Step-response specific costs (always computed for metrics)
        cost_abs = self.w_abs * torch.abs(e_theta)
        cost_time = self.w_time * (~inside_band).to(torch.float32)
        cost_osc = self.w_osc * osc_event
        cost_overshoot = self.w_overshoot * (overshoot_excess**2)

        # Settling bonus (one-time)
        settle_bonus = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        if torch.any(just_settled) and float(self.settle_time_target_s) > 0.0:
            speed_factor = torch.clamp(
                1.0 - (self._settle_time_s / float(self.settle_time_target_s)), min=0.0
            )
            settle_bonus = torch.where(
                just_settled, float(self.w_settle_bonus) * speed_factor, settle_bonus
            )

        # Compute reward based on mode
        if self.reward_mode == "tracking":
            # Universal tracking reward: only base LQR-like cost + progress shaping
            # Step-response metrics computed but NOT included in reward
            cost_total = cost_base
            reward = (-cost_total) * float(self.reward_scale) + progress
        else:  # "step_response"
            # Full reward with step-specific penalties
            cost_total = cost_base + cost_abs + cost_time + cost_osc + cost_overshoot
            reward = (-cost_total) * float(self.reward_scale) + progress + settle_bonus

        # Update action history
        self.pre_prev_action = self.prev_action
        self.prev_action = u_norm

        # Termination/truncation
        terminated = torch.abs(theta) > float(self.max_pitch_rad)
        truncated = self.step_count >= int(self.number_time_steps - 2)
        done = terminated | truncated
        # --------------------------------------------------------------
        # Survival shaping (optional):
        # - discourage early termination
        # - encourage finishing the full time horizon
        # --------------------------------------------------------------
        term_val = float(-100.0 - float(self.early_termination_penalty))
        if self.early_termination_penalty_per_step != 0.0:
            remaining = torch.clamp(
                (int(self.number_time_steps - 2) - self.step_count).to(torch.float32),
                min=0.0,
            )
            term_val = (
                term_val - float(self.early_termination_penalty_per_step) * remaining
            )
            reward = torch.where(terminated, term_val, reward)
        else:
            reward = torch.where(terminated, torch.full_like(reward, term_val), reward)
        if self.survival_bonus != 0.0:
            reward = reward + float(self.survival_bonus) * (~terminated).to(
                torch.float32
            )
        if self.completion_bonus != 0.0:
            reward = reward + float(self.completion_bonus) * (
                truncated & (~terminated)
            ).to(torch.float32)

        if self.auto_reset and torch.any(done):
            self._reset_done(done)

        obs = self._get_obs()
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info
