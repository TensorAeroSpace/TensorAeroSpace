"""Critic network for IHDP.

This module defines the Critic component used by the IHDP agent.
"""

import math
import random
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn


def _activation_from_string(name: str) -> nn.Module:
    """Convert an activation function name to a PyTorch module.

    Args:
        name: Name of the activation function.

    Returns:
        Corresponding PyTorch activation module.
    """
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "linear" or name is None:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class Critic:
    """Provides Critic class with function approximator (NN) for Critic class.

    Critic creates neural network model using PyTorch and can train network online.
    User can choose number of layers, number of neurons, batch size, number of epochs and activation functions.

    Args:
        Q_weights: Q-function weights.
        selected_states: Selected states.
        tracking_states: Tracked states.
        indices_tracking_states: Index of tracked states.
        number_time_steps: Number of time steps.
        start_training: Training start step.
        gamma (float, optional): Gamma discount factor. Defaults to 0.8.
        learning_rate (int, optional): Learning rate. Defaults to 2.
        learning_rate_exponent_limit (int, optional): Learning rate exponent limit. Defaults to 10.
        layers (tuple, optional): Number of layers and neurons in layers. Defaults to (10, 6, 1).
        activations (tuple, optional): Activation functions in layers. Defaults to ("sigmoid", "sigmoid", "linear").
        WB_limits (int, optional): Weight value constraints. Defaults to 30.
        NN_initial (optional): Initial weight values. Defaults to None.
        model_path (optional): Model path. Defaults to None.
    """

    # Class attributes
    # Attributes related to RMSprop
    beta_rmsprop = 0.999
    epsilon = 1e-8

    # Attributes related to the momentum
    beta_momentum = 0.9

    def __init__(
        self,
        Q_weights: list[float],
        selected_states: list[str],
        tracking_states: list[str],
        indices_tracking_states: list[int],
        number_time_steps: int,
        start_training: int,
        gamma: float = 0.8,
        learning_rate: float = 2,
        learning_rate_exponent_limit: int = 10,
        layers: tuple[int, ...] = (10, 6, 1),
        activations: tuple[str, ...] = ("sigmoid", "sigmoid", "linear"),
        WB_limits: float = 30,
        NN_initial: int | None = None,
        model_path: str | None = None,
    ) -> None:
        """Initialize IHDP Critic network and buffers.

        Args:
            Q_weights: Diagonal weights for Q matrix.
            selected_states: State variable names.
            tracking_states: Tracked states for cost.
            indices_tracking_states: Indices of tracked states.
            number_time_steps: Total steps in episode.
            start_training: Step index to start training.
            gamma: Discount factor.
            learning_rate: Optimizer learning rate.
            learning_rate_exponent_limit: Exponent limit for LR decay.
            layers: Hidden layer sizes.
            activations: Activation functions per layer.
            WB_limits: Weight/bias clipping limit.
            NN_initial: Optional weight initializer seed.
            model_path: Optional path to load/save model.
        """
        # Declaration of attributes regarding the states and rewards
        self.number_states = len(selected_states)
        self.number_tracking_states = len(tracking_states)
        self.indices_tracking_states = indices_tracking_states
        self.xt = None
        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt_ref = None
        self.xt_ref_1 = np.zeros((self.number_tracking_states, 1))
        self.ct = 0
        self.ct_1 = 0
        self.Jt = 0
        self.Jt_1 = 0
        self.model_path = model_path
        if len(Q_weights) < self.number_tracking_states:
            raise Exception("The size of Q_weights needs to equal the number of states")
        self.Q = np.zeros((self.number_tracking_states, self.number_tracking_states))
        np.fill_diagonal(self.Q, Q_weights)
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.start_training = start_training

        # Store the states
        self.store_states = np.zeros((self.number_time_steps, self.number_states, 1))

        # Declaration of attributes related to the neural network
        if layers[-1] != 1:
            raise Exception("The last layer should have a single neuron.")
        elif len(layers) != len(activations):
            raise Exception(
                "The number of layers needs to be equal to the number of activations."
            )
        self.layers = layers
        self.activations = activations
        self.model = None
        self.dJt_dxt = None
        self.NN_initial = NN_initial

        # Declaration of attributes related to the cost function
        if not (0 <= gamma <= 1):
            raise Exception("The forgetting factor should be in the range [0,1]")
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.learning_rate_exponent_limit = learning_rate_exponent_limit
        self.WB_limits = WB_limits
        self.store_J = np.zeros((1, self.number_time_steps))
        self.store_J_1 = np.zeros((1, self.number_time_steps))
        self.store_c = np.zeros((1, self.number_time_steps))

        # Declaration of the storage arrays for the weights
        self.store_weights = {}

        # Attributes related to the momentum
        self.momentum_dict = {}

        # Attributes related to RMSprop
        self.rmsprop_dict = {}

        # Attributes related to experience replay
        self.replay = []

    def save_model(self):
        """Save model."""
        torch.save(self.model.state_dict(), "./critic_weight.pt")

    def load_model(self):
        """Load weights."""
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

    def save_Jt_ct(self):
        """Save critic state evaluation."""
        np.save("./critic_jt", [self.Jt_1, self.Jt, self.ct_1, self.ct])

    def load_Jt_ct(self):
        """Load critic state evaluation."""
        data = np.load("./critic_jt.npy", allow_pickle=True)
        self.Jt_1 = data[0]
        self.Jt = data[1]
        self.ct_1 = data[2]
        self.ct = data[3]

    def build_critic_model(self):
        """Function creating neural network. Currently this is a densely connected neural network. User can
        define number of layers, number of neurons, and activation function.
        """
        if self.NN_initial is not None:
            torch.manual_seed(self.NN_initial)

        modules = []
        modules.append(nn.Flatten(start_dim=1))

        # First layer: input is number_tracking_states
        in_features = self.number_tracking_states
        linear = nn.Linear(in_features, self.layers[0])
        # VarianceScaling(scale=1, mode='fan_in', distribution='truncated_normal')
        # fan_in for Linear is in_features, std = sqrt(scale / fan_in)
        std = math.sqrt(1.0 / in_features)
        nn.init.trunc_normal_(linear.weight, mean=0.0, std=std)
        nn.init.zeros_(linear.bias)
        modules.append(linear)
        modules.append(_activation_from_string(self.activations[0]))

        self.store_weights["W1"] = np.zeros(
            (self.number_tracking_states * self.layers[0], self.number_time_steps + 1)
        )
        self.store_weights["W1"][:, self.time_step] = (
            linear.weight.detach().numpy().flatten()
        )

        for counter, layer in enumerate(self.layers[1:]):
            in_feat = self.layers[counter]
            out_feat = self.layers[counter + 1]
            linear = nn.Linear(in_feat, out_feat)
            std = math.sqrt(1.0 / in_feat)
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=std)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            modules.append(_activation_from_string(self.activations[counter + 1]))

            self.store_weights["W" + str(counter + 2)] = np.zeros(
                (
                    self.layers[counter] * self.layers[counter + 1],
                    self.number_time_steps + 1,
                )
            )
            self.store_weights["W" + str(counter + 2)][:, self.time_step] = (
                linear.weight.detach().numpy().flatten()
            )

        self.model = nn.Sequential(*modules)
        self.model.eval()

        # Initialize momentum and rmsprop dicts for all parameters
        for count in range(len(list(self.model.parameters()))):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0

    def run_train_critic_online_adaptive_alpha(
        self, xt: np.ndarray, xt_ref: np.ndarray
    ) -> np.ndarray:
        """Function that evaluates critic neural network once and returns J(xt) value. At the same
        time it trains function approximator with adaptive learning rate scheme.

        Args:
            xt: Current state of time step.
            xt_ref: Reference state of current time step for computing one-step cost function.

        Returns:
            Jt: Critic evaluation at current time step.
        """

        nn_input, dJt_dW = self.compute_forward_pass(xt, xt_ref)
        dE_dJ, ec_critic_before, EC_critic_before = self.compute_loss_derivative()

        params = list(self.model.parameters())
        weight_cache = [p.data.clone() for p in params]

        network_improvement = False
        n_reductions = 0
        while not network_improvement and self.time_step > self.start_training:
            for count in range(len(dJt_dW)):
                update = dE_dJ * dJt_dW[count]
                params[count].data -= torch.tensor(
                    np.reshape(
                        self.learning_rate * update,
                        params[count].shape,
                    ),
                    dtype=torch.float32,
                )

                # Implement WB_limits
                self.check_WB_limits(count)

            updated_Jt = self.model(nn_input).detach().numpy()
            ec_critic_after = (
                np.reshape(-self.ct_1 - self.gamma * updated_Jt, [-1, 1]) + self.Jt_1
            )
            Ec_critic_after = 0.5 * np.square(ec_critic_after)

            # In the case that the error is not decreased, the time step is repeated with half the learning rate
            if Ec_critic_after <= EC_critic_before or n_reductions > 10:
                network_improvement = True
                # The learning rate is doubled if the network errors have the same signs
                if np.sign(ec_critic_before) == np.sign(ec_critic_after):
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
                for WB_count in range(len(params)):
                    params[WB_count].data.copy_(weight_cache[WB_count])

        return self.Jt

    def run_train_critic_online_adam(
        self, xt: np.ndarray, xt_ref: np.ndarray
    ) -> np.ndarray:
        """Function that evaluates critic neural network once and returns J(xt) value. At the same
        time, it trains function approximator using Adam optimizer.

        Args:
            xt: Current state of time step.
            xt_ref: Reference state of current time step for computing one-step cost function.

        Returns:
            Jt: Critic evaluation at current time step.
        """

        # Safe the information in the replay attribute
        self.replay.append((self.xt_1, xt, self.ct_1))

        # Obtain the forward pass of the critic and the derivatives of the output with respect to the weights and biases
        nn_input, dJt_dW = self.compute_forward_pass(xt, xt_ref)

        # Obtain the derivative of the loss with respect to the critic NN output (Jt)
        dE_dJ, _, _ = self.compute_loss_derivative()

        # Run the Adam optimizer given the gradients
        self.adam_iteration(dJt_dW, dE_dJ)

        return self.Jt

    def adam_iteration(self, dJt_dW: list[np.ndarray], dE_dJ: np.ndarray) -> None:
        """Adam updates all weights and biases considering loss function derivative with respect to NN
        output and derivative of neural network output with respect to weights and biases.

        Args:
            dJt_dW (_type_): Derivative of NN output with respect to weights and biases.
            dE_dJ (_type_): Derivative of loss function with respect to NN output.
        """

        if self.time_step > self.start_training:
            params = list(self.model.parameters())
            for count in range(len(dJt_dW)):
                gradient = dE_dJ * dJt_dW[count]
                momentum = (
                    self.beta_momentum * self.momentum_dict[count]
                    + (1 - self.beta_momentum) * gradient
                )
                self.momentum_dict[count] = momentum
                momentum_corrected = momentum / (
                    1 - self.beta_momentum ** (self.time_step + 1)
                )

                rmsprop = self.beta_rmsprop * self.rmsprop_dict[count] + (
                    1 - self.beta_rmsprop
                ) * np.multiply(gradient, gradient)
                self.rmsprop_dict[count] = rmsprop
                rmsprop_corrected = rmsprop / (
                    1 - self.beta_rmsprop ** (self.time_step + 1)
                )

                update = momentum_corrected / (
                    np.sqrt(rmsprop_corrected) + self.epsilon
                )

                params[count].data -= torch.tensor(
                    np.reshape(
                        self.learning_rate * update,
                        params[count].shape,
                    ),
                    dtype=torch.float32,
                )

                # Implement WB_limits
                self.check_WB_limits(count)

                if count % 2 == 1:
                    params[count].data.zero_()

            # Update the learning rate
            self.learning_rate = max(self.learning_rate * 0.995, 0.000001)

    def run_train_critic_online_alpha_decay(
        self, xt: np.ndarray, xt_ref: np.ndarray
    ) -> np.ndarray:
        """Evaluate the critic once and update it with a decaying learning rate.

        Args:
            xt: Current state.
            xt_ref: Reference state used to compute one-step cost.

        Returns:
            np.ndarray: Critic value estimate at the current time step.
        """

        # Safe the information in the replay attribute
        self.replay.append((self.xt_1, xt, self.ct_1))

        # Obtain the forward pass of the critic and the derivatives of the output with respect to the weights and biases
        nn_input, dJt_dW = self.compute_forward_pass(xt, xt_ref)
        nn_input_1, dJt_dW_1, _ = self.compute_forward_pass(
            self.xt_1, self.xt_ref_1, replay=True
        )

        # Obtain the derivative of the loss with respect to the critic NN output (Jt)
        dE_dJ, ec_critic_before, _ = self.compute_loss_derivative()
        dE_dJ = ec_critic_before

        if self.time_step > self.start_training:
            params = list(self.model.parameters())
            for count in range(len(dJt_dW_1)):
                gradient = dE_dJ * dJt_dW_1[count]
                params[count].data -= torch.tensor(
                    np.reshape(
                        self.learning_rate * gradient,
                        params[count].shape,
                    ),
                    dtype=torch.float32,
                )
                # Implement WB_limits
                self.check_WB_limits(count)

                if count % 2 == 1:
                    params[count].data.zero_()

            # Update the learning rate
            self.learning_rate = max(self.learning_rate * 0.995, 0.000001)

        return self.Jt

    def train_critic_replay_adam(self, replay_size: int, iteration: int) -> None:
        """Train the critic using samples from the replay buffer (Adam)."""

        # Compute the number of data points used in the replay training
        replay_size = min(replay_size, len(self.replay))

        # Define the data points that are going to be used in the replay training
        indices = list(range(len(self.replay)))
        random.shuffle(indices)
        for i in range(replay_size):
            # Extract the data point information
            index = indices[i]
            replay = self.replay[index]

            xt_1, xt_ref_1, xt, xt_ref, ct_1 = replay
            tracked_states = np.reshape(xt_1[self.indices_tracking_states, :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref_1, [-1, 1])
            nn_input_1 = torch.tensor(
                np.array([xt_error]).astype("float32"), dtype=torch.float32
            )

            # Obtain the forward pass of xt and the derivative of the output with respect to weights and biases
            nn_input, dJt_dW, Jt = self.compute_forward_pass(xt, xt_ref, replay=True)

            # Obtain the forward pass of xt_1
            with torch.no_grad():
                Jt_1 = self.model(nn_input_1).numpy()

            # Obtain the derivative of the critic cost function with respect to the critic output
            dE_dJ, _, _ = self.compute_loss_derivative(Jt_1, Jt, ct_1)

            # Carry out the Adam optimisation
            self.adam_iteration(dJt_dW, dE_dJ, iteration)

    def compute_forward_pass(
        self, xt: np.ndarray, xt_ref: np.ndarray, replay: bool = False
    ) -> Tuple[Any, list[np.ndarray]]:
        """Compute critic output and gradients with respect to weights/biases."""
        # If it is online, safe the input in the object
        if not replay:
            self.xt = xt
            self.xt_ref = xt_ref
            self.ct = self.c_computation()
            if self.time_step == 0:
                self.ct_1 = self.ct

        # Define the input to the critic NN
        # Check if xt already contains only tracked states
        if xt.shape[0] == len(self.indices_tracking_states):
            tracked_states = np.reshape(xt, [-1, 1])
        else:
            tracked_states = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
        xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])

        nn_input = torch.tensor(
            np.array([xt_error]).astype("float32"), dtype=torch.float32
        )

        # Run the input through the network watching the weights and biases for later derivatives
        params = list(self.model.parameters())
        # Enable grad for parameters temporarily
        for p in params:
            p.requires_grad_(True)

        prediction = self.model(nn_input)

        # Obtain the derivative of the output with respect to the weights and biases
        dJt_dW_tensors = torch.autograd.grad(
            prediction, params, create_graph=False, retain_graph=False
        )
        dJt_dW = [g.detach().numpy() for g in dJt_dW_tensors]

        # Disable grad for parameters after computing gradients
        for p in params:
            p.requires_grad_(False)

        # In the case that it is online, safe the output in the object; otherwise provide as function output
        if not replay:
            self.Jt = prediction.detach().numpy()
            self.store_J[:, self.time_step] = np.reshape(self.Jt, [-1])
            return nn_input, dJt_dW
        else:
            Jt = prediction.detach().numpy()
            return nn_input, dJt_dW, Jt

    def compute_loss_derivative(
        self, *args: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute derivative of the critic loss with respect to critic output."""

        # In the case that there are no inputs, obtain data from the object attributes
        if len(args) == 0:
            # Check if xt_1 already contains only tracked states
            if self.xt_1.shape[0] == len(self.indices_tracking_states):
                tracked_states = np.reshape(self.xt_1, [-1, 1])
            else:
                tracked_states = np.reshape(
                    self.xt_1[self.indices_tracking_states, :], [-1, 1]
                )
            xt_1_error = np.reshape(tracked_states - self.xt_ref_1, [-1, 1])
            nn_input_1 = torch.tensor(
                np.array([xt_1_error]).astype("float32"), dtype=torch.float32
            )

            with torch.no_grad():
                self.Jt_1 = self.model(nn_input_1).numpy()
            Jt = self.Jt
            target = self.targets_computation_online()
        elif len(args) == 3:
            self.Jt_1 = args[0]
            Jt = args[1]
            ct_1 = args[2]
            target = self.targets_computation_online(Jt, ct_1)
        else:
            self.Jt_1 = 0
            Jt = 0
            target = 0
            Exception("Unexpected number of arguments.")

        # Compute the network error
        ec_critic_before = target + self.Jt_1
        self.store_J_1[:, self.time_step] = np.reshape(self.Jt_1, [-1])

        # Compute the derivative of the loss function with respect to the critic network output (Jt)
        dE_dJ = -self.gamma * ec_critic_before

        # Check what is the critic and actor loss values before the critic network update.
        EC_critic_before = 0.5 * np.square(ec_critic_before)

        return dE_dJ, ec_critic_before, EC_critic_before

    def check_WB_limits(self, count):
        """Clamp weights/biases to the configured absolute limit (WB_limits)."""
        params = list(self.model.parameters())
        params[count].data.clamp_(-self.WB_limits, self.WB_limits)

    def evaluate_critic(
        self, xt: np.ndarray, xt_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate critic and compute gradient with respect to the input."""

        xt = np.asarray(xt).squeeze()
        if xt.ndim == 0:
            xt = xt.reshape(1)
        xt = xt.reshape(-1, 1)
        if xt.shape[0] <= len(self.indices_tracking_states):
            tracked_states = xt.reshape(-1, 1)
        else:
            tracked_states = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
        xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
        nn_input = torch.tensor(
            np.array([xt_error]).astype("float32"),
            dtype=torch.float32,
            requires_grad=True,
        )

        prediction = self.model(nn_input)

        Jt = prediction.detach().numpy()
        dJt_dxt = torch.autograd.grad(prediction, nn_input)[0].detach().numpy()

        return Jt, dJt_dxt

    def c_computation(self) -> np.ndarray:
        """Compute one-step cost for the current time step."""

        # Check if xt already contains only tracked states
        if self.xt.shape[0] == len(self.indices_tracking_states):
            tracked_states = self.xt
        else:
            tracked_states = self.xt[self.indices_tracking_states, :]

        ct = np.matmul(
            np.matmul((np.reshape(tracked_states, [-1, 1]) - self.xt_ref).T, self.Q),
            (np.reshape(tracked_states, [-1, 1]) - self.xt_ref),
        )
        self.store_c[0, self.time_step] = ct.flat[0]
        return ct

    def targets_computation_online(self, *args: Any) -> np.ndarray:
        """Compute the TD target used for critic training."""

        if len(args) == 0:
            target = np.reshape(-self.ct_1 - self.gamma * self.Jt, [-1, 1])
        elif len(args) == 2:
            Jt = args[0]
            ct_1 = args[1]
            target = np.reshape(-ct_1 - self.gamma * Jt, [-1, 1])
        else:
            Exception("Unexpected number of arguments")
            target = 0
        return target

    def update_critic_attributes(self) -> None:
        """Update time-dependent critic attributes after each step."""
        self.time_step += 1
        self.ct_1 = self.ct
        self.xt_1 = self.xt
        self.xt_ref_1 = self.xt_ref

        # Store the weights
        params = list(self.model.parameters())
        for counter in range(len(self.layers)):
            # In the Sequential, each layer block is: Flatten, [Linear, Activation, ...]
            # Parameters are ordered as: weight_0, bias_0, weight_1, bias_1, ...
            # counter * 2 gives the weight parameter index (skipping biases)
            self.store_weights["W" + str(counter + 1)][:, self.time_step] = (
                params[counter * 2].detach().numpy().flatten()
            )

    def restart_time_step(self) -> None:
        """Reset the time step counter to zero."""
        self.time_step = 0

    def restart_critic(self) -> None:
        """Reset critic internal state and buffers."""
        # Declaration of attributes regarding the states and rewards
        self.time_step = 0
        self.xt = None
        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt_ref = None
        self.xt_ref_1 = np.zeros((self.number_tracking_states, 1))
        self.ct = 0
        self.ct_1 = 0
        self.Jt = 0
        self.Jt_1 = 0
        self.learning_rate = self.learning_rate_0

        # Store the states
        self.store_states = np.zeros((self.number_time_steps, self.number_states, 1))

        # Declaration of attributes related to the neural network
        self.dJt_dxt = None

        # Declaration of attributes related to the cost function
        self.store_J = np.zeros((1, self.number_time_steps))
        self.store_c = np.zeros((1, self.number_time_steps))

        # Restart momentum and rmsprop
        for count in range(len(list(self.model.parameters()))):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0
