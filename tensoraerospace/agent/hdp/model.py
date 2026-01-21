# flake8: noqa
"""Model-based Heuristic Dynamic Programming (HDP) agent.

This class exposes the HDP configuration of `tensoraerospace.agent.ADP` as a
dedicated, self-documenting entry point. All training and evaluation logic is
delegated to the ADP implementation with ``design="hdp"`` (model-based critic
J(R), actor improved via one-step lookahead with known linearized dynamics).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch

from tensoraerospace.agent.adp.adp import ADP


class HDP(ADP):
    """Heuristic Dynamic Programming (HDP) agent — model-based Adaptive Critic Design.

    HDP is a model-based reinforcement learning algorithm from the Adaptive Critic
    Designs (ACD) family. It uses a known linearized system model (matrices A, B)
    to perform one-step lookahead for actor improvement. The critic network learns
    a scalar cost-to-go function J(R), while the actor is optimized by backpropagating
    through the model to minimize expected future cost.

    The algorithm follows the framework from Prokhorov & Wunsch (1997):
      - Critic learns: J(R_t) ≈ U_t + γ J(R_{t+1})
      - Actor minimizes: U_t + γ J(A·R_t + B·π(R_t)) via model-based lookahead

    Example:
        >>> import numpy as np
        >>> from tensoraerospace.agent.hdp import HDP
        >>> from tensoraerospace.envs.b747 import ImprovedB747Env
        >>>
        >>> def step_reference(steps, deg=5.0):
        ...     ref = np.zeros((1, steps), dtype=np.float32)
        ...     ref[:, steps // 5:] = np.deg2rad(deg)
        ...     return ref
        >>>
        >>> env = ImprovedB747Env(
        ...     initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        ...     reference_signal=step_reference(800, deg=5.0),
        ...     number_time_steps=800,
        ...     dt=0.02,
        ... )
        >>> agent = HDP(env, gamma=0.99, hidden_size=256)
        >>> agent.train(num_episodes=100)

    References:
        - Prokhorov D.V., Wunsch D.C. "Adaptive Critic Designs."
          IEEE Trans. Neural Networks, vol. 8, no. 5, pp. 997-1007, 1997.
        - Werbos P.J. "Approximate dynamic programming for real-time control
          and neural modeling." Handbook of Intelligent Control, 1992.

    Attributes:
        actor: Neural network that outputs control action π(R).
        critic: Neural network that estimates cost-to-go J(R).
        env: The Gymnasium-compatible environment.
        gamma: Discount factor for future costs.

    Note:
        HDP requires an environment with:
          - ``env.model.filt_A``, ``env.model.filt_B``: linearized system matrices
          - ``env.reference_signal``: pitch reference trajectory array
    """

    def __init__(
        self,
        env: Any,
        *,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_size: int = 256,
        device: Union[str, torch.device] = "cpu",
        seed: int = 42,
        exploration_std: float = 0.1,
        dhp_w_theta: float = 5.0,
        dhp_w_q: float = 0.2,
        dhp_w_u: float = 0.01,
        dhp_w_du: float = 0.02,
        dhp_use_env_cost: bool = True,
        dhp_use_baseline: bool = False,
        dhp_baseline_type: str = "pd",
        dhp_baseline_kp: float = 0.6,
        dhp_baseline_ki: float = 0.0,
        dhp_baseline_kd: float = 0.2,
        dhp_pid_use_normalized_theta: bool = True,
        dhp_pid_mode: str = "norm",
        dhp_residual_scale: float = 1.0,
        dhp_warmstart_actor_episodes: int = 0,
        dhp_warmstart_actor_epochs: int = 2,
        dhp_warmstart_actor_disable_baseline_after: bool = True,
        dhp_critic_cycle_episodes: int = 0,
        dhp_action_cycle_episodes: int = 0,
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize the HDP agent.

        Args:
            env: Gymnasium-compatible environment. Must provide:
                - ``env.model.filt_A``, ``env.model.filt_B``: linearized dynamics matrices
                - ``env.reference_signal``: reference trajectory for tracking
                - ``env.observation_space``, ``env.action_space``: Box spaces

            gamma: Discount factor for future costs. Controls the trade-off between
                immediate and future costs. Range: [0, 1]. Default: 0.99.

            actor_lr: Learning rate for the actor network optimizer (Adam).
                Default: 3e-4.

            critic_lr: Learning rate for the critic network optimizer (Adam).
                Default: 3e-4.

            hidden_size: Number of neurons in each hidden layer of both actor and
                critic networks. Both networks use two hidden layers with Tanh
                activation. Default: 256.

            device: Torch device for computation ('cpu', 'cuda', 'mps', or
                torch.device instance). Default: 'cpu'.

            seed: Random seed for reproducibility. Affects PyTorch, NumPy, and
                exploration noise. Default: 42.

            exploration_std: Standard deviation of Gaussian noise added to actions
                during training for exploration. Set to 0 for deterministic training.
                Default: 0.1.

            dhp_w_theta: Weight for pitch angle tracking error in the cost function.
                Higher values prioritize pitch tracking accuracy. Default: 5.0.

            dhp_w_q: Weight for pitch rate tracking error in the cost function.
                Default: 0.2.

            dhp_w_u: Weight for control magnitude penalty in the cost function.
                Penalizes large control inputs. Default: 0.01.

            dhp_w_du: Weight for control rate (smoothness) penalty in the cost
                function. Penalizes rapid changes in control. Default: 0.02.

            dhp_use_env_cost: If True, use cost weights from the environment
                (e.g., ImprovedB747Env.w_pitch) when available. If False or
                unavailable, use dhp_w_* parameters. Default: True.

            dhp_use_baseline: If True, use a PD/PID baseline controller and train
                the actor as a residual policy: u = u_baseline + scale * π(R).
                Helps stabilize training in early stages. Default: False.

            dhp_baseline_type: Type of baseline controller: 'pd' (proportional-
                derivative) or 'pid' (with integral term). Default: 'pd'.

            dhp_baseline_kp: Proportional gain for the baseline controller.
                Default: 0.6.

            dhp_baseline_ki: Integral gain for the baseline PID controller.
                Only used if dhp_baseline_type='pid'. Default: 0.0.

            dhp_baseline_kd: Derivative gain for the baseline controller.
                Default: 0.2.

            dhp_pid_use_normalized_theta: If True, normalize pitch angle by
                max_pitch_rad before passing to PID baseline. Default: True.

            dhp_pid_mode: PID computation mode: 'norm' (normalized angles) or
                'deg' (degrees). Default: 'norm'.

            dhp_residual_scale: Scaling factor for the learned residual policy
                when using baseline. Final action: u_baseline + scale * π(R).
                Default: 1.0.

            dhp_warmstart_actor_episodes: Number of episodes to pre-train the
                actor by imitating the baseline controller via supervised learning.
                This initializes the actor as a stabilizing controller before
                ACD updates begin (paper recommendation). Default: 0 (disabled).

            dhp_warmstart_actor_epochs: Number of supervised learning epochs per
                warm-start episode. Default: 2.

            dhp_warmstart_actor_disable_baseline_after: If True, disable the
                baseline (set dhp_use_baseline=False) after warm-start completes,
                so the actor takes full control. Default: True.

            dhp_critic_cycle_episodes: Number of episodes to train only the critic
                (actor frozen) in each cycle. Part of the alternating training
                schedule from Prokhorov & Wunsch Section III. Set to 0 to disable
                alternating and train both networks every step. Default: 0.

            dhp_action_cycle_episodes: Number of episodes to train only the actor
                (critic frozen) in each cycle. Works with dhp_critic_cycle_episodes
                for alternating training. Default: 0.

            log_dir: Directory path for TensorBoard logs. If None, logging is
                disabled. Default: None.

            log_every_updates: Frequency of logging (every N gradient updates).
                Default: 100.

            **kwargs: Additional arguments passed to the base ADP class.

        Raises:
            ValueError: If gamma is not in [0, 1].
            ValueError: If exploration_std is negative.
            ValueError: If environment lacks required attributes (filt_A, filt_B,
                reference_signal).
        """
        # Ensure caller-provided design flag does not override the dedicated HDP setup.
        kwargs.pop("design", None)
        super().__init__(
            env,
            design="hdp",
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            hidden_size=hidden_size,
            device=device,
            seed=seed,
            exploration_std=exploration_std,
            use_replay=False,
            use_target_networks=False,
            log_dir=log_dir,
            log_every_updates=log_every_updates,
            dhp_w_theta=dhp_w_theta,
            dhp_w_q=dhp_w_q,
            dhp_w_u=dhp_w_u,
            dhp_w_du=dhp_w_du,
            dhp_use_env_cost=dhp_use_env_cost,
            dhp_use_baseline=dhp_use_baseline,
            dhp_baseline_type=dhp_baseline_type,
            dhp_baseline_kp=dhp_baseline_kp,
            dhp_baseline_ki=dhp_baseline_ki,
            dhp_baseline_kd=dhp_baseline_kd,
            dhp_pid_use_normalized_theta=dhp_pid_use_normalized_theta,
            dhp_pid_mode=dhp_pid_mode,
            dhp_residual_scale=dhp_residual_scale,
            dhp_warmstart_actor_episodes=dhp_warmstart_actor_episodes,
            dhp_warmstart_actor_epochs=dhp_warmstart_actor_epochs,
            dhp_warmstart_actor_disable_baseline_after=dhp_warmstart_actor_disable_baseline_after,
            dhp_critic_cycle_episodes=dhp_critic_cycle_episodes,
            dhp_action_cycle_episodes=dhp_action_cycle_episodes,
            **kwargs,
        )
