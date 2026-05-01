"""Actor network for IHDP.

This module defines the Actor component used by the IHDP agent.
"""

import glob
import math
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn


def _activation_from_string(name: str) -> nn.Module:
    """Convert an activation function name string to a PyTorch module.

    Args:
        name: Activation name ('sigmoid', 'tanh', 'relu', 'linear').

    Returns:
        Corresponding ``nn.Module`` activation layer.

    Raises:
        ValueError: If the activation name is not recognised.
    """
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")


def _build_sequential(
    input_dim: int,
    layers: tuple,
    activations: tuple,
    seed: int,
) -> nn.Sequential:
    """Build an ``nn.Sequential`` model for the actor network.

    Weight initialisation uses variance-scaled truncated normal
    ``trunc_normal_(std=sqrt(0.01 / fan_in))``; biases are zero-initialised.

    Args:
        input_dim: Number of input features.
        layers: Tuple of neuron counts per layer.
        activations: Tuple of activation names per layer.
        seed: Random seed for reproducibility.

    Returns:
        A ``nn.Sequential`` model.
    """
    torch.manual_seed(seed)

    modules = []
    in_features = input_dim
    for i, (neurons, act_name) in enumerate(zip(layers, activations)):
        linear = nn.Linear(in_features, neurons)
        # VarianceScaling(scale=0.01, mode='fan_in', distribution='truncated_normal')
        fan_in = in_features
        std = math.sqrt(0.01 / fan_in)
        nn.init.trunc_normal_(linear.weight, mean=0.0, std=std)
        nn.init.zeros_(linear.bias)
        modules.append(linear)
        modules.append(_activation_from_string(act_name))
        in_features = neurons

    model = nn.Sequential(*modules)
    model.eval()  # Default to eval mode (no dropout / batchnorm)
    return model


def _get_trainable_parameters(model: nn.Sequential) -> list[torch.Tensor]:
    """Return trainable parameters in a stable order.

    For each ``nn.Linear`` layer the order is ``[weight, bias]``.

    Args:
        model: The ``nn.Sequential`` model.

    Returns:
        Ordered list of parameter tensors.
    """
    params = []
    for module in model:
        if isinstance(module, nn.Linear):
            params.append(module.weight)
            params.append(module.bias)
    return params


class Actor:
    """Actor Model in IHDP.

    Provides Actor class with Actor function approximator (NN).
    Actor creates neural network model using PyTorch and can train network online.
    User can choose number of layers, number of neurons, batch size, number of epochs and activation functions.

    Args:
        selected_inputs: Selected control signals.
        selected_states: Selected state signals.
        tracking_states: Tracked states.
        indices_tracking_states: Indices of tracked states.
        number_time_steps: Number of time steps.
        start_training: Step from which training begins.
        layers (tuple, optional): Model layers. Defaults to (6, 1).
        activations (tuple, optional): Activation layers ('sigmoid', 'sigmoid').
        learning_rate (float, optional): Learning rate. Defaults to 0.9.
        learning_rate_cascaded (float, optional): Learning rate in cascade mode. Defaults to 0.9.
        learning_rate_exponent_limit (int, optional): Learning rate exponent limit. Defaults to 10.
        type_PE (str, optional): PE type. Defaults to '3211'.
        amplitude_3211 (int, optional): 3211 amplitude. Defaults to 1.
        pulse_length_3211 (int, optional): 3211 pulse length. Defaults to 15.
        WB_limits (int, optional): Weight limits. Defaults to 30.
        maximum_input (int, optional): Maximum value. Defaults to 25.
        maximum_q_rate (int, optional): Maximum rate. Defaults to 20.
        cascaded_actor (bool, optional): Enable cascade network mode. Defaults to False.
        NN_initial (optional): Weight initialization. Defaults to None.
        cascade_tracking_state (list, optional): Tracking in cascade mode. Defaults to ['alpha', 'wz'].
        model_path (str, optional): Model path for loading weights. Defaults to None.
    """

    beta_rmsprop = 0.999
    epsilon = 1e-8

    # Attributes related to the momentum
    beta_momentum = 0.9

    def __init__(
        self,
        selected_inputs: list[str],
        selected_states: list[str],
        tracking_states: list[str],
        indices_tracking_states: list[int],
        number_time_steps: int,
        start_training: int,
        layers: tuple[int, ...] = (6, 1),
        activations: tuple[str, ...] = ("sigmoid", "sigmoid"),
        learning_rate: float = 0.9,
        learning_rate_cascaded: float = 0.9,
        learning_rate_exponent_limit: int = 10,
        type_PE: str = "3211",
        amplitude_3211: float = 1,
        pulse_length_3211: int = 15,
        WB_limits: float = 30,
        maximum_input: float = 25,
        maximum_q_rate: float = 20,
        cascaded_actor: bool = False,
        NN_initial: int | None = None,
        cascade_tracking_state: list[str] | None = None,
        model_path: str | None = None,
        use_integral_correction: bool = False,
        integral_gain: float = 0.0,
        integral_clamp_deg: float = 5.0,
        integral_warmup_steps: int = 500,
    ) -> None:
        """Initialize IHDP Actor network and hyperparameters.

        Args:
            selected_inputs: Control input names.
            selected_states: State variable names.
            tracking_states: Tracked states for reward.
            indices_tracking_states: Indices of tracked states in state vector.
            number_time_steps: Total time steps in episode.
            start_training: Step index to start training.
            layers: Hidden layer sizes.
            activations: Activations per layer.
            learning_rate: Base learning rate.
            learning_rate_cascaded: Learning rate for cascaded mode.
            learning_rate_exponent_limit: Exponent limit for LR decay.
            type_PE: Persistent excitation pattern.
            amplitude_3211: Amplitude for 3211 signal.
            pulse_length_3211: Pulse length for 3211 signal.
            WB_limits: Weight/bias clipping limit.
            maximum_input: Max control magnitude.
            maximum_q_rate: Max pitch rate.
            cascaded_actor: Whether to use cascaded network.
            NN_initial: Optional weight initializer seed.
            cascade_tracking_state: Tracking states for cascade mode.
            model_path: Path to load/save model weights.
            use_integral_correction: Enable an integral compensation term
                added to the actor's control output. IHDP minimizes a
                quadratic LQ-functional and therefore has no integral
                action — the closed loop has a small but persistent
                steady-state offset on setpoint-tracking. Enabling this
                flag adds ``-K_I · ∫(ref − y) dτ`` to ``u`` (with
                anti-windup clipping) which removes the offset without
                touching the actor architecture or the existing IHDP
                training loop.
            integral_gain: ``K_I`` (units: ``deg / (rad·s)`` of integral
                state on the ``stab`` channel). Tuned empirically; values
                in the range 10..25 work well for the F-16 alpha-tracking
                benchmark. Ignored if ``use_integral_correction`` is False.
            integral_clamp_deg: Anti-windup limit on the integrator
                state, in degrees. Caps the magnitude of accumulated
                error so a long initial transient cannot saturate the
                actuator after settling. Default ``5°``.
            integral_warmup_steps: Number of steps after which the
                integrator starts accumulating. Set this to be at least
                as long as the persistent-excitation pulse, so the
                integrator does not "see" the PE injection as a real
                tracking error. Default ``500`` (==5 s at dt=0.01s).
        """
        self.number_inputs = len(selected_inputs)
        self.selected_states = selected_states
        self.cascade_tracking_state = (
            cascade_tracking_state
            if cascade_tracking_state is not None
            else ["alpha", "wz"]
        )
        self.number_states = len(selected_states)
        self.number_tracking_states = len(tracking_states)
        self.indices_tracking_states = indices_tracking_states
        self.xt = None
        self.xt_ref = None
        self.ut = 0
        self.maximum_input = maximum_input
        self.maximum_q_rate = maximum_q_rate
        self.model_path = model_path
        # Attributes related to time
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.start_training = start_training

        # Attributes related to the NN
        self.model = None
        self.model_q = None
        if layers[-1] != 1:
            raise Exception("The last layer should have a single neuron.")
        elif len(layers) != len(activations):
            raise Exception(
                "The number of layers needs to be equal to the number of activations."
            )
        self.layers = layers
        self.activations = activations
        self.learning_rate = learning_rate
        self.learning_rate_cascaded = learning_rate_cascaded
        self.learning_rate_0 = learning_rate
        self.learning_rate_exponent_limit = learning_rate_exponent_limit
        self.WB_limits = WB_limits
        self.NN_initial = NN_initial

        # Attributes related to the persistent excitation
        self.type_PE = type_PE
        self.amplitude_3211 = amplitude_3211
        self.pulse_length_3211 = pulse_length_3211

        # Attributes related to the training of the NN
        self.dut_dWb = None
        self.dut_dWb_1 = None

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

        # Attributes related to the momentum
        self.momentum_dict = {}

        # Attributes related to RMSprop
        self.rmsprop_dict = {}

        # Declaration of the storage arrays for the weights
        self.store_weights = {}
        self.store_weights_q = {}

        # Attributes for the cascaded actor
        self.cascaded_actor = cascaded_actor
        self.dut_dq_ref = None
        self.dq_ref_dWb = None
        self.store_q = np.zeros((1, self.number_time_steps))

        # Integral-correction layer — kills the residual steady-state
        # offset of the LQR-style policy. State is held inside the actor
        # so :meth:`restart_actor` resets it together with the other
        # episode-level attributes.
        self.use_integral_correction = bool(use_integral_correction)
        self.integral_gain = float(integral_gain)
        self.integral_clamp = math.radians(float(integral_clamp_deg))
        self.integral_warmup_steps = int(integral_warmup_steps)
        self.integral_state = 0.0

    def build_actor_model(self):
        """Function creating Actor network. This is a fully connected network.
        Can define number of layers, number of neurons per layer, and activation functions.

        In cascade mode the actor decomposes into two SISO sub-networks:
            * outer ``model``    : alpha_error (scalar)  -> q_ref
            * inner ``model_q``  : q_error     (scalar)  -> u

        Both sub-networks therefore have ``input_dim=1`` regardless of the
        number of tracking states declared on the agent. The cascade
        ``indices_tracking_states`` rewrite (which advertises ``[alpha_idx,
        wz_idx]`` so that :meth:`run_actor_online` can address both rows of
        the augmented observation) is applied AFTER both sub-models have
        been built — otherwise ``create_NN`` would size ``model_q`` for a
        2-element input and matrix multiplication would fail at runtime.
        """

        # First Neural Network — outer (alpha_error -> q_ref). Built while
        # ``indices_tracking_states`` is still single-element so input_dim=1.
        self.model, self.store_weights = self.create_NN(self.store_weights, 120)

        # Second Neural Network for the cascaded actor (q_error -> u).
        # Same single-element indices view is needed here so that
        # ``create_NN`` produces input_dim=1.
        if self.cascaded_actor:
            print("It is assumed that the input to the NNs is the tracking error.")
            self.model_q, self.store_weights_q = self.create_NN(
                self.store_weights_q, 120
            )

            # Now switch to the cascade-aware indices so that
            # ``run_actor_online`` / ``evaluate_actor`` can index both the
            # primary tracked state (alpha) and the inner-loop state (wz)
            # in the augmented observation vector returned by the env.
            tracking_states = self.cascade_tracking_state
            self.indices_tracking_states = [
                self.selected_states.index(tracking_states[i])
                for i in range(len(tracking_states))
            ]
            self.number_tracking_states = len(tracking_states)

        params = _get_trainable_parameters(self.model)
        n_params = len(params)
        if self.cascaded_actor:
            n_params += len(_get_trainable_parameters(self.model_q))
        for count in range(n_params):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0

    def save_model(self):
        """Save model."""
        torch.save(self.model.state_dict(), "actor_weight.pt")

    def save_dut_dWb(self):
        """Save gradient."""
        for i in range(len(self.dut_dWb)):
            np.save(f"./actor_dut_dWb/{i}_dut_dWb.txt", self.dut_dWb[i])

    def load_dut_dWb(self):
        """Load gradient."""
        line = []
        for file in sorted(glob.glob("./actor_dut_dWb/*")):
            line.append(np.load(file, allow_pickle=False))
        self.dut_dWb = line
        self.dut_dWb_1 = line

    def load_model(self):
        """Load model weights."""
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

    def create_NN(self, store_weights: dict, seed: int) -> Tuple[nn.Sequential, dict]:
        """Create NN with user input.

        Args:
            store_weights: Dictionary containing weights and biases.
            seed: Seed for saving random variables.

        Returns:
            model: Created NN model.
            store_weights: Dictionary containing updated weights and biases.

        """

        # Determine input dimension based on number of tracked states
        input_dim = (
            len(self.indices_tracking_states)
            if hasattr(self, "indices_tracking_states")
            else 1
        )

        model = _build_sequential(input_dim, self.layers, self.activations, seed)
        params = _get_trainable_parameters(model)

        # Store initial weights (kernel / weight matrices only, matching TF convention)
        # In TF: trainable_variables[0] = W1, [2] = W2, ...  (even indices are kernels)
        # params[0] = W1 (weight), params[1] = b1 (bias), params[2] = W2, ...
        store_weights["W1"] = np.zeros(
            (input_dim * self.layers[0], self.number_time_steps + 1)
        )
        # Note: PyTorch stores weight as (out_features, in_features), TF as (in, out).
        # We flatten the same way (just flatten the whole matrix) for storage.
        store_weights["W1"][:, self.time_step] = params[0].detach().numpy().flatten()

        for counter, layer in enumerate(self.layers[1:]):
            store_weights["W" + str(counter + 2)] = np.zeros(
                (
                    self.layers[counter] * self.layers[counter + 1],
                    self.number_time_steps + 1,
                )
            )
            store_weights["W" + str(counter + 2)][:, self.time_step] = (
                params[(counter + 1) * 2].detach().numpy().flatten()
            )

        return model, store_weights

    def _forward_with_grad(
        self, model: nn.Sequential, nn_input: torch.Tensor
    ) -> torch.Tensor:
        """Run a forward pass ensuring all parameters require grad and input has grad.

        Args:
            model: The nn.Sequential model.
            nn_input: Input tensor.

        Returns:
            Output tensor from the model (with grad_fn attached).
        """
        # Ensure parameters require grad
        for p in model.parameters():
            p.requires_grad_(True)
        return model(nn_input)

    def run_actor_online(self, xt: np.ndarray, xt_ref: np.ndarray) -> np.ndarray:
        """Generate system input with given and real states.

        Args:
            xt: Current state of time step.
            xt_ref: Reference state of current time step.

        Returns:
            ut: Input to system and incremental model.
        """

        if self.cascaded_actor:
            self.xt = xt
            self.xt_ref = xt_ref

            # Check if xt already contains only tracked states
            if xt.shape[0] == len(self.indices_tracking_states):
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states[0], :], [-1, 1]
                )
            else:
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states[0], :], [-1, 1]
                )
            alphat_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input_alpha = torch.tensor(
                np.array([alphat_error]).astype("float32"), requires_grad=False
            )

            # Forward pass through model (alpha -> q_ref) with gradient tracking
            for p in self.model.parameters():
                p.requires_grad_(True)
            q_ref_tensor = self.model(nn_input_alpha)
            # Compute dq_ref/dWb
            params = _get_trainable_parameters(self.model)
            grads = torch.autograd.grad(
                q_ref_tensor, params, create_graph=False, retain_graph=False
            )
            self.dq_ref_dWb = [g.detach().numpy() for g in grads]

            q_ref_val = q_ref_tensor.detach().numpy()
            if self.activations[-1] == "sigmoid":
                q_ref = max(
                    min(
                        (2 * self.maximum_q_rate * q_ref_val) - self.maximum_q_rate,
                        np.reshape(self.maximum_q_rate, q_ref_val.shape),
                    ),
                    np.reshape(-self.maximum_q_rate, q_ref_val.shape),
                )
            elif self.activations[-1] == "tanh":
                q_ref = max(
                    min(
                        (self.maximum_q_rate * q_ref_val),
                        np.reshape(self.maximum_q_rate, q_ref_val.shape),
                    ),
                    np.reshape(-self.maximum_q_rate, q_ref_val.shape),
                )

            self.store_q[:, self.time_step] = q_ref

            # Check if xt already contains only tracked states
            if xt.shape[0] == len(self.indices_tracking_states):
                tracked_states_q = np.reshape(
                    xt[self.indices_tracking_states[1], :], [-1, 1]
                )
            else:
                tracked_states_q = np.reshape(
                    xt[self.indices_tracking_states[1], :], [-1, 1]
                )
            qt_error = np.reshape(
                tracked_states_q - np.reshape(q_ref, tracked_states_q.shape), [-1, 1]
            )
            nn_input_q = torch.tensor(
                np.array([qt_error]).astype("float32"), requires_grad=True
            )

            # Forward pass through model_q (q_error -> ut) - get dut/dq_ref (via input grad)
            for p in self.model_q.parameters():
                p.requires_grad_(True)
            ut_tensor = self.model_q(nn_input_q)
            # dut/d(nn_input_q) = dut/dq_ref
            grad_input = torch.autograd.grad(
                ut_tensor, nn_input_q, create_graph=False, retain_graph=True
            )
            self.dut_dq_ref = grad_input[0].detach().numpy()

            # dut/dWb_q
            params_q = _get_trainable_parameters(self.model_q)
            grads_q = torch.autograd.grad(
                ut_tensor, params_q, create_graph=False, retain_graph=False
            )
            self.dut_dWb = [g.detach().numpy() for g in grads_q]

            ut = ut_tensor.detach().numpy()

        else:
            self.xt = xt
            self.xt_ref = xt_ref

            # If xt already contains only tracked states, use it directly
            if xt.shape[0] == len(self.indices_tracking_states):
                tracked_states = np.reshape(xt, [-1, 1])
            else:
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states, :], [-1, 1]
                )
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            # Create input data with correct dimension for model
            nn_input = torch.tensor(
                xt_error.flatten().reshape(1, -1).astype("float32"),
                requires_grad=False,
            )

            for p in self.model.parameters():
                p.requires_grad_(True)
            ut_tensor = self.model(nn_input)
            params = _get_trainable_parameters(self.model)
            grads = torch.autograd.grad(
                ut_tensor,
                params,
                create_graph=False,
                retain_graph=False,
            )
            self.dut_dWb = [g.detach().numpy() for g in grads]

            ut = ut_tensor.detach().numpy()

        e0 = self.compute_persistent_excitation()

        if self.activations[-1] == "sigmoid":
            self.ut = max(
                min(
                    (2 * self.maximum_input * ut) - self.maximum_input + e0,
                    np.reshape(self.maximum_input, ut.shape),
                ),
                np.reshape(-self.maximum_input, ut.shape),
            )
        elif self.activations[-1] == "tanh":
            ut = max(
                min(
                    (self.maximum_input * ut),
                    np.reshape(self.maximum_input, ut.shape),
                ),
                np.reshape(-self.maximum_input, ut.shape),
            )

            self.ut = max(
                min(ut + e0, np.reshape(self.maximum_input, ut.shape)),
                np.reshape(-self.maximum_input, ut.shape),
            )

        # Ensure we return array, not scalar
        if np.isscalar(self.ut):
            return np.array([self.ut])
        return self.ut

    def train_actor_online(
        self, Jt1: np.ndarray, dJt1_dxt1: np.ndarray, G: np.ndarray
    ) -> None:
        """Get chain rule elements, calculate gradient and apply it to corresponding weights and biases.

        Args:
            Jt1 (_type_): dEa/dJ
            dJt1_dxt1 (_type_): dJ/dx
            G: dx/du, obtained from incremental model.
        """

        Jt1 = Jt1.flatten()[0]

        chain_rule = Jt1 * np.matmul(
            np.reshape(G[self.indices_tracking_states, :], [-1, 1]).T, dJt1_dxt1
        )

        chain_rule = chain_rule.flatten()[0]
        params = _get_trainable_parameters(self.model)
        for count in range(len(self.dut_dWb)):
            update = chain_rule * self.dut_dWb[count]
            with torch.no_grad():
                params[count] -= torch.tensor(
                    np.reshape(
                        self.learning_rate * update,
                        params[count].shape,
                    ).astype(np.float32)
                )

            # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
            self._check_WB_limits_param(params[count])

    def train_actor_online_adaptive_alpha(
        self,
        Jt1: np.ndarray,
        dJt1_dxt1: np.ndarray,
        G: np.ndarray,
        incremental_model: Any,
        critic: Any,
        xt_ref1: np.ndarray,
    ) -> None:
        """Train Actor using adaptive alpha depending on sign and magnitude of network errors.

        Args:
            Jt1: Critic evaluation with incremental model next time step prediction.
            dJt1_dxt1: Critical network gradient with respect to incremental model next time prediction.
            G: Input data distribution matrix.
            incremental_model: Incremental model.
            critic: Critic.
            xt_ref1: Reference state at next time step.
        """
        Ec_actor_before = 0.5 * np.square(Jt1)
        params = _get_trainable_parameters(self.model)
        weight_cache = [p.detach().clone() for p in params]
        network_improvement = False
        n_reductions = 0
        while not network_improvement and self.time_step > self.start_training:
            # Train the actor
            self.train_actor_online(Jt1, dJt1_dxt1, G)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)

            # Code for checking whether the learning rate of the actor should be halved
            if Ec_actor_after <= Ec_actor_before or n_reductions > 10:
                network_improvement = True
                if np.sign(Jt1) == np.sign(Jt1_after):
                    self.learning_rate = min(
                        2 * self.learning_rate,
                        self.learning_rate_0 * 2**self.learning_rate_exponent_limit,
                    )
            else:
                n_reductions += 1
                self.learning_rate = max(
                    self.learning_rate / 2,
                    self.learning_rate_0 / 2**self.learning_rate_exponent_limit,
                )
                with torch.no_grad():
                    for WB_count, p in enumerate(params):
                        p.copy_(weight_cache[WB_count])

    def train_actor_online_adam(
        self,
        Jt1: np.ndarray,
        dJt1_dxt1: np.ndarray,
        G: np.ndarray,
        incremental_model: Any,
        critic: Any,
        xt_ref1: np.ndarray,
    ) -> None:
        """Train the actor online using Adam updates."""
        if self.cascaded_actor:
            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(
                np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1
            )

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training and np.abs(self.ut) < 25:
                for count in range(len(self.dut_dWb)):
                    if self.activations[-1] == "sigmoid":
                        gradient = (
                            2 * self.maximum_input * chain_rule * self.dut_dWb[count]
                        )
                    elif self.activations[-1] == "tanh":
                        gradient = self.maximum_input * chain_rule * self.dut_dWb[count]
                    else:
                        raise Exception(
                            "There is no code for the defined output activation function."
                        )
                    (
                        self.model_q,
                        self.learning_rate_cascaded,
                    ) = self.compute_Adam_update(
                        count, gradient, self.model_q, self.learning_rate_cascaded
                    )

                for count in range(len(self.dq_ref_dWb)):
                    if self.activations[-1] == "sigmoid":
                        gradient = (
                            -2
                            * self.maximum_q_rate
                            * chain_rule
                            * self.dut_dq_ref
                            * self.dq_ref_dWb[count]
                        )
                    elif self.activations[-1] == "tanh":
                        gradient = (
                            -self.maximum_q_rate
                            * chain_rule
                            * self.dut_dq_ref
                            * self.dq_ref_dWb[count]
                        )
                    else:
                        raise Exception(
                            "There is no code for the defined output activation function."
                        )

                    self.model, self.learning_rate = self.compute_Adam_update(
                        count, gradient, self.model, self.learning_rate
                    )
            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
        else:
            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(
                np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1
            )

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training:
                for count in range(len(self.dut_dWb)):
                    gradient = chain_rule * self.dut_dWb[count]
                    self.model, self.learning_rate = self.compute_Adam_update(
                        count, gradient, self.model, self.learning_rate
                    )

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)

    def train_actor_online_alpha_decay(
        self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1
    ):
        """Train the actor with a learning rate that decays over time."""

        if self.cascaded_actor:
            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(
                np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1
            )

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training and np.abs(self.ut) < 25:
                params_q = _get_trainable_parameters(self.model_q)
                for count in range(len(self.dut_dWb)):
                    if self.activations[-1] == "sigmoid":
                        gradient = (
                            2 * self.maximum_input * chain_rule * self.dut_dWb[count]
                        )
                    elif self.activations[-1] == "tanh":
                        gradient = self.maximum_input * chain_rule * self.dut_dWb[count]
                    with torch.no_grad():
                        params_q[count] -= torch.tensor(
                            np.reshape(
                                self.learning_rate_cascaded * gradient,
                                params_q[count].shape,
                            ).astype(np.float32)
                        )

                    # Implement WB_limits
                    self._check_WB_limits_param(params_q[count])
                    if count % 2 == 1:
                        with torch.no_grad():
                            params_q[count].zero_()

                params = _get_trainable_parameters(self.model)
                for count in range(len(self.dq_ref_dWb)):
                    if self.activations[-1] == "sigmoid":
                        gradient = (
                            -2
                            * self.maximum_q_rate
                            * chain_rule
                            * self.dut_dq_ref
                            * self.dq_ref_dWb[count]
                        )
                    elif self.activations[-1] == "tanh":
                        gradient = (
                            -self.maximum_q_rate
                            * chain_rule
                            * self.dut_dq_ref
                            * self.dq_ref_dWb[count]
                        )
                    with torch.no_grad():
                        params[count] -= torch.tensor(
                            np.reshape(
                                self.learning_rate * gradient,
                                params[count].shape,
                            ).astype(np.float32)
                        )
                    self._check_WB_limits_param(params[count])
                    if count % 2 == 1:
                        with torch.no_grad():
                            params[count].zero_()

                # Update the learning rate
                self.learning_rate = max(self.learning_rate * 0.9995, 0.0001)
                self.learning_rate_cascaded = max(
                    self.learning_rate_cascaded * 0.9995, 0.0001
                )
            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
        else:
            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(
                np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1
            )

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training:
                params = _get_trainable_parameters(self.model)
                for count in range(len(self.dut_dWb)):
                    gradient = chain_rule * self.dut_dWb[count]
                    with torch.no_grad():
                        params[count] -= torch.tensor(
                            np.reshape(
                                self.learning_rate * gradient,
                                params[count].shape,
                            ).astype(np.float32)
                        )
                    # Implement WB_limits
                    self._check_WB_limits_param(params[count])
                    if count % 2 == 1:
                        with torch.no_grad():
                            params[count].zero_()
                # Update the learning rate
                self.learning_rate = max(self.learning_rate * 0.995, 0.001)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)

    def compute_Adam_update(
        self,
        count: int,
        gradient: np.ndarray,
        model: nn.Sequential,
        learning_rate: float,
    ) -> Tuple[nn.Sequential, float]:
        """Compute an Adam-style weight update and apply it."""

        momentum = (
            self.beta_momentum * self.momentum_dict[count]
            + (1 - self.beta_momentum) * gradient
        )
        self.momentum_dict[count] = momentum
        momentum_corrected = momentum / (
            1 - np.power(self.beta_momentum, self.time_step + 1)
        )

        rmsprop = self.beta_rmsprop * self.rmsprop_dict[count] + (
            1 - self.beta_rmsprop
        ) * np.multiply(gradient, gradient)
        self.rmsprop_dict[count] = rmsprop
        rmsprop_corrected = rmsprop / (
            1 - np.power(self.beta_rmsprop, self.time_step + 1)
        )

        update = momentum_corrected / (np.sqrt(rmsprop_corrected) + self.epsilon)

        params = _get_trainable_parameters(model)
        with torch.no_grad():
            params[count] -= torch.tensor(
                np.reshape(learning_rate * update, params[count].shape).astype(
                    np.float32
                )
            )
        # Implement WB_limits
        self._check_WB_limits_param(params[count])
        if count % 2 == 1:
            with torch.no_grad():
                params[count].zero_()

        if count == len(params) - 1:
            learning_rate = max(learning_rate * 0.9995, 0.0001)

        return model, learning_rate

    def check_WB_limits(self, count: int, model: nn.Sequential) -> nn.Sequential:
        """Clamp weights/biases that exceed the configured WB_limits."""
        params = _get_trainable_parameters(model)
        with torch.no_grad():
            params[count].clamp_(-self.WB_limits, self.WB_limits)
        return model

    def _check_WB_limits_param(self, param: torch.Tensor) -> None:
        """Clamp a single parameter tensor in-place to [-WB_limits, WB_limits]."""
        with torch.no_grad():
            param.clamp_(-self.WB_limits, self.WB_limits)

    def compute_persistent_excitation(self, *args: int) -> float:
        """Compute the persistent excitation term for the current time step."""
        if len(args) == 1:
            t = args[0] + 1
        elif len(args) == 0:
            t = self.time_step + 1

        e0_1 = 0
        e0_2 = 0
        if self.type_PE == "sinusoidal" or self.type_PE == "combined":
            e0_1 = (
                np.sin(t)
                * np.cos(2 * t)
                * (np.sin(3 * t + np.pi / 4) + np.cos(4 * t - np.pi / 3))
                * 1e-2
            )

        if self.type_PE == "3211" or self.type_PE == "combined":
            if t < 3 * self.pulse_length_3211 / 7:
                e0_2 = 0.5 * self.amplitude_3211
            elif t < 5 * self.pulse_length_3211 / 7:
                e0_2 = -0.5 * self.amplitude_3211
            elif t < 6 * self.pulse_length_3211 / 7:
                e0_2 = 0.8 * self.amplitude_3211
            elif t < self.pulse_length_3211:
                e0_2 = -self.amplitude_3211

        e0 = e0_1 + e0_2

        return float(e0)

    def update_actor_attributes(self) -> None:
        """Update time-dependent actor attributes after each time step."""
        self.time_step += 1
        self.dut_dWb_1 = self.dut_dWb

        params = _get_trainable_parameters(self.model)
        for counter in range(len(self.layers)):
            self.store_weights["W" + str(counter + 1)][:, self.time_step] = (
                params[counter * 2].detach().numpy().flatten()
            )

        if self.cascaded_actor:
            params_q = _get_trainable_parameters(self.model_q)
            for counter in range(len(self.layers)):
                self.store_weights_q["W" + str(counter + 1)][:, self.time_step] = (
                    params_q[counter * 2].detach().numpy().flatten()
                )

    def evaluate_actor(self, *args: Any) -> np.ndarray:
        """Evaluate the actor using provided or stored state/reference."""
        if len(args) == 0:
            xt = self.xt
            xt_ref = self.xt_ref
        elif len(args) == 1:
            xt = self.xt
            xt_ref = self.xt_ref
            time_step = args[0]
        elif len(args) == 2:
            xt = args[0]
            xt_ref = args[1]
        else:
            raise Exception("THERE SHOULD BE AN OUTPUT in the evaluate_actor function.")

        if self.cascaded_actor:
            # Check if xt already contains only tracked states
            if xt.shape[0] == len(self.indices_tracking_states):
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states[0], :], [-1, 1]
                )
            else:
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states[0], :], [-1, 1]
                )
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = torch.tensor(np.array([xt_error]).astype("float32"))

            with torch.no_grad():
                q_ref_0 = self.model(nn_input).numpy()
            if self.activations[-1] == "sigmoid":
                q_ref = max(
                    min(
                        (2 * self.maximum_q_rate * q_ref_0) - self.maximum_q_rate,
                        np.reshape(self.maximum_q_rate, q_ref_0.shape),
                    ),
                    np.reshape(-self.maximum_q_rate, q_ref_0.shape),
                )
            elif self.activations[-1] == "tanh":
                q_ref = max(
                    min(
                        (self.maximum_q_rate * q_ref_0),
                        np.reshape(self.maximum_q_rate, q_ref_0.shape),
                    ),
                    np.reshape(-self.maximum_q_rate, q_ref_0.shape),
                )

            # Check if xt already contains only tracked states
            if xt.shape[0] == len(self.indices_tracking_states):
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states[1], :], [-1, 1]
                )
            else:
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states[1], :], [-1, 1]
                )
            xt_error_q = np.reshape(
                tracked_states - np.reshape(q_ref, tracked_states.shape), [-1, 1]
            )
            nn_input_q = torch.tensor(np.array([xt_error_q]).astype("float32"))

            with torch.no_grad():
                ut = self.model_q(nn_input_q).numpy()

        else:
            # Check if xt already contains only tracked states
            if xt.shape[0] == len(self.indices_tracking_states):
                tracked_states = np.reshape(xt, [-1, 1])
            else:
                tracked_states = np.reshape(
                    xt[self.indices_tracking_states, :], [-1, 1]
                )
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = torch.tensor(np.array([xt_error]).astype("float32"))

            with torch.no_grad():
                ut = self.model(nn_input).numpy()

        if len(args) == 1:
            e0 = self.compute_persistent_excitation(time_step)
        else:
            e0 = self.compute_persistent_excitation()

        if self.activations[-1] == "sigmoid":
            ut = max(
                min(
                    (2 * self.maximum_input * ut) - self.maximum_input + e0,
                    np.reshape(self.maximum_input, ut.shape),
                ),
                np.reshape(-self.maximum_input, ut.shape),
            )
        elif self.activations[-1] == "tanh":
            ut = max(
                min(
                    ((self.maximum_input + 10) * ut) + e0,
                    np.reshape(self.maximum_input, ut.shape),
                ),
                np.reshape(-self.maximum_input, ut.shape),
            )
        return ut

    def restart_time_step(self):
        """Reset the internal time-step counter to zero."""
        self.time_step = 0

    def restart_actor(self):
        """Reset actor state and optimizer-related attributes."""
        self.time_step = 0
        self.xt = None
        self.xt_ref = None
        self.ut = 0

        # Attributes related to the training of the NN
        self.dut_dWb = None
        self.dut_dWb_1 = None
        self.learning_rate = self.learning_rate_0

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

        # Reset the integral-correction state on a fresh episode.
        self.integral_state = 0.0

        # Restart momentum and rmsprop
        params = _get_trainable_parameters(self.model)
        for count in range(len(params)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0
