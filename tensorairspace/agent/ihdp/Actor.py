#!/usr/bin/env python
"""Provides class Actor with the function approximator (NN) of the Actor

Actor creates the Neural Network model with Tensorflow and it can train the network online.
The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions.
"""

import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

"----------------------------------------------------------------------------------------------------------------------"
__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright (C) 2020 Jose Ignacio"
__credits__ = []
__license__ = "MIT"
__version__ = "2.0.1"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Production"
"----------------------------------------------------------------------------------------------------------------------"


class Actor:
    # Class attributes
    # Attributes related to RMSprop
    beta_rmsprop = 0.999
    epsilon = 1e-8

    # Attributes related to the momentum
    beta_momentum = 0.9

    def __init__(self, selected_inputs, selected_states, tracking_states, indices_tracking_states,
                 number_time_steps, start_training, layers=(6, 1), activations=('sigmoid', 'sigmoid'),
                 learning_rate=0.9, learning_rate_cascaded=0.9, learning_rate_exponent_limit=10,
                 type_PE='3211', amplitude_3211=1, pulse_length_3211=15, WB_limits=30,
                 maximum_input=25, maximum_q_rate=20, cascaded_actor=False, NN_initial=None,
                 cascade_tracking_state=['alpha', 'wz'], model_path: str = None):
        self.number_inputs = len(selected_inputs)
        self.selected_states = selected_states
        self.cascade_tracking_state = cascade_tracking_state
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
            raise Exception("The number of layers needs to be equal to the number of activations.")
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

    def build_actor_model(self):
        """
        Function that creates the actor ANN architecture. It is a densely connected neural network. The user
        can decide the number of layers, the number of neurons, as well as the activation function.
        :return:
        """

        # First Neural Network
        self.model, self.store_weights = self.create_NN(self.store_weights, 120)

        # Second Neural Network for the cascaded actor
        if self.cascaded_actor:
            print("It is assumed that the input to the NNs is the tracking error.")
            tracking_states = self.cascade_tracking_state
            self.indices_tracking_states = [self.selected_states.index(tracking_states[i]) for i in
                                            range(len(tracking_states))]
            self.number_tracking_states = len(tracking_states)

            self.model_q, self.store_weights_q = self.create_NN(self.store_weights_q, 120)

        for count in range(len(self.model.trainable_variables) * 2):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0

    def save_model(self):
        self.model.save_weights("actor_weight.h5")

    def save_dut_dWb(self):
        for i in range(len(self.dut_dWb)):
            np.save(f"./actor_dut_dWb/{i}_dut_dWb.txt", self.dut_dWb[i])

    def load_dut_dWb(self):
        line = []
        for file in glob.glob('./actor_dut_dWb/*'):
            line.append(tf.constant((np.load(file, allow_pickle=True))))
        self.dut_dWb = line
        self.dut_dWb_1 = line

    def load_model(self):
        self.model.load_weights(self.model_path)

    def create_NN(self, store_weights, seed):
        """
        Creates a NN given the user input
        :param store_weights: dictionary containing weights and biases
        :return: model --> the created NN model
                store_weights --> the dictionary that contains the updated weights and biases.
        """
        # initializer = tf.keras.initializers.GlorotNormal()
        initializer = tf.keras.initializers.VarianceScaling(
            scale=0.01, mode='fan_in', distribution='truncated_normal', seed=seed)
        model = tf.keras.Sequential()

        model.add(Flatten(input_shape=(1, 1), name='Flatten_1'))

        model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                        name='dense_1'))

        store_weights['W1'] = np.zeros((1 * self.layers[0], self.number_time_steps + 1))
        store_weights['W1'][:, self.time_step] = model.trainable_variables[0].numpy().flatten()

        for counter, layer in enumerate(self.layers[1:]):
            model.add(Dense(self.layers[counter + 1], activation=self.activations[counter + 1],
                            kernel_initializer=initializer, name='dense_' + str(counter + 2)))
            store_weights['W' + str(counter + 2)] = np.zeros(
                (self.layers[counter] * self.layers[counter + 1], self.number_time_steps + 1))
            store_weights['W' + str(counter + 2)][:, self.time_step] = model.trainable_variables[
                (counter + 1) * 2].numpy().flatten()

        return model, store_weights

    def run_actor_online(self, xt, xt_ref):
        """
        Generate input to the system with the reference and real states.
        :param xt: current time_step states
        :param xt_ref: current time step reference states
        :return: ut --> input to the system and the incremental model
        """

        if self.cascaded_actor:
            self.xt = xt
            self.xt_ref = xt_ref

            tracked_states = np.reshape(xt[self.indices_tracking_states[0], :], [-1, 1])
            alphat_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input_alpha = tf.constant(np.array([(alphat_error)]).astype('float32'))

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                q_ref = self.model(nn_input_alpha)
            self.dq_ref_dWb = tape.gradient(q_ref, self.model.trainable_variables)

            if self.activations[-1] == 'sigmoid':
                q_ref = max(min((2 * self.maximum_q_rate * q_ref.numpy()) - self.maximum_q_rate,
                                np.reshape(self.maximum_q_rate, q_ref.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref.numpy().shape))
            elif self.activations[-1] == 'tanh':
                q_ref = max(min((self.maximum_q_rate * q_ref.numpy()),
                                np.reshape(self.maximum_q_rate, q_ref.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref.numpy().shape))

            self.store_q[:, self.time_step] = q_ref

            tracked_states_q = np.reshape(xt[self.indices_tracking_states[1], :], [-1, 1])
            qt_error = np.reshape(tracked_states_q - np.reshape(q_ref, tracked_states_q.shape), [-1, 1])
            nn_input_q = tf.constant(np.array([(qt_error)]).astype('float32'))

            with tf.GradientTape() as tape:
                tape.watch(nn_input_q)
                ut = self.model_q(nn_input_q)

            self.dut_dq_ref = tape.gradient(ut, nn_input_q)

            with tf.GradientTape() as tape:
                tape.watch(self.model_q.trainable_variables)
                ut = self.model_q(nn_input_q)

            self.dut_dWb = tape.gradient(ut, self.model_q.trainable_variables)

        else:
            self.xt = xt
            self.xt_ref = xt_ref

            tracked_states = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = tf.constant(np.array([(xt_error)]).astype('float32'))

            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                ut = self.model(nn_input)
            self.dut_dWb = tape.gradient(ut, self.model.trainable_variables)

        e0 = self.compute_persistent_excitation()

        if self.activations[-1] == 'sigmoid':
            self.ut = max(min((2 * self.maximum_input * ut.numpy()) - self.maximum_input + e0,
                              np.reshape(self.maximum_input, ut.numpy().shape)),
                          np.reshape(-self.maximum_input, ut.numpy().shape))
        elif self.activations[-1] == 'tanh':
            ut = max(min((self.maximum_input * ut.numpy()),
                         np.reshape(self.maximum_input, ut.numpy().shape)),
                     np.reshape(-self.maximum_input, ut.numpy().shape))

            self.ut = max(min(ut + e0,
                              np.reshape(self.maximum_input, ut.shape)),
                          np.reshape(-self.maximum_input, ut.shape))

        return self.ut

    def train_actor_online(self, Jt1, dJt1_dxt1, G):
        """
        Obtains the elements of the chain rule, computes the gradient and applies it to the corresponding weights and
        biases.
        :param Jt1: dEa/dJ
        :param dJt1_dxt1: dJ/dx
        :param G: dx/du, obtained from the incremental model
        :return:
        """
        Jt1 = Jt1.flatten()[0]

        chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states, :], [-1, 1]).T, dJt1_dxt1)

        chain_rule = chain_rule.flatten()[0]
        for count in range(len(self.dut_dWb)):
            update = chain_rule * self.dut_dWb[count]
            self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update,
                                                                        self.model.trainable_variables[count].shape))

            # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
            self.model = self.check_WB_limits(count, self.model)

    def train_actor_online_adaptive_alpha(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """
        Train the actor with an adaptive alpha depending on the sign and magnitude of the network errors
        :param Jt1: the evaluation of the critic with the next time step prediction of the incremental model
        :param dJt1_dxt1: the gradient of the critic network with respect to the next time prediction of the incremental model
        :param G: the input distribution matrix
        :param incremental_model: the incremental model
        :param critic: the critic
        :param xt_ref1: reference states at the next time step
        :return:
        """
        Ec_actor_before = 0.5 * np.square(Jt1)
        # print("ACTOR LOSS xt1 before= ", Ec_actor_before)
        weight_cache = [tf.Variable(self.model.trainable_variables[i].numpy()) for i in
                        range(len(self.model.trainable_variables))]
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
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)

            # Code for checking whether the learning rate of the actor should be halved
            if Ec_actor_after <= Ec_actor_before or n_reductions > 10:
                network_improvement = True
                if np.sign(Jt1) == np.sign(Jt1_after):
                    self.learning_rate = min(2 * self.learning_rate,
                                             self.learning_rate_0 * 2 ** self.learning_rate_exponent_limit)
                    # print("ACTOR LEARNING_RATE = ", self.learning_rate)
            else:
                n_reductions += 1
                self.learning_rate = max(self.learning_rate / 2,
                                         self.learning_rate_0 / 2 ** self.learning_rate_exponent_limit)
                for WB_count in range(len(self.model.trainable_variables)):
                    self.model.trainable_variables[WB_count].assign(weight_cache[WB_count].numpy())
                # print("ACTOR LEARNING_RATE = ", self.learning_rate)

    def train_actor_online_adam(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """
        Train the actor with the Adam optimizer .
        :param Jt1: the evaluation of the critic with the next time step prediction of the incremental model
        :param dJt1_dxt1: the gradient of the critic network with respect to the next time prediction of the incremental model
        :param G: the input distribution matrix
        :param incremental_model: the incremental model
        :param critic: the critic
        :param xt_ref1: reference states at the next time step
        :return:
        """
        if self.cascaded_actor:
            Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training and np.abs(self.ut) < 25:
                for count in range(len(self.dut_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = 2 * self.maximum_input * chain_rule * self.dut_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = self.maximum_input * chain_rule * self.dut_dWb[count]
                    else:
                        raise Exception("There is no code for the defined output activation function.")
                    self.model_q, self.learning_rate_cascaded = self.compute_Adam_update(count, gradient,
                                                                                         self.model_q,
                                                                                         self.learning_rate_cascaded)

                for count in range(len(self.dq_ref_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = -2 * self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = -self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    else:
                        raise Exception("There is no code for the defined output activation function.")

                    self.model, self.learning_rate = self.compute_Adam_update(count, gradient,
                                                                              self.model, self.learning_rate)
            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)
        else:
            Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training:
                for count in range(len(self.dut_dWb)):
                    gradient = chain_rule * self.dut_dWb[count]
                    self.model, self.learning_rate = self.compute_Adam_update(count, gradient,
                                                                              self.model, self.learning_rate)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)

    def train_actor_online_alpha_decay(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """
        Train the actor with a learning rate that decay with the number of time steps
        :param Jt1: the evaluation of the critic with the next time step prediction of the incremental model
        :param dJt1_dxt1: the gradient of the critic network with respect to the next time prediction of the incremental model
        :param G: the input distribution matrix
        :param incremental_model: the incremental model
        :param critic: the critic
        :param xt_ref1: reference states at the next time step
        :return:
        """
        if self.cascaded_actor:
            Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training and np.abs(self.ut) < 25:
                for count in range(len(self.dut_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = 2 * self.maximum_input * chain_rule * self.dut_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = self.maximum_input * chain_rule * self.dut_dWb[count]
                    self.model_q.trainable_variables[count].assign_sub(
                        np.reshape(self.learning_rate_cascaded * gradient,
                                   self.model_q.trainable_variables[
                                       count].shape))

                    # Implement WB_limits: the weights and biases can not have values whose absolute value
                    # exceeds WB_limits
                    self.model_q = self.check_WB_limits(count, self.model_q)
                    if count % 2 == 1:
                        self.model_q.trainable_variables[count].assign(
                            np.zeros(self.model_q.trainable_variables[count].shape))
                for count in range(len(self.dq_ref_dWb)):
                    if self.activations[-1] == 'sigmoid':
                        gradient = -2 * self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    elif self.activations[-1] == 'tanh':
                        gradient = -self.maximum_q_rate * chain_rule * self.dut_dq_ref * self.dq_ref_dWb[count]
                    self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * gradient,
                                                                                self.model.trainable_variables[
                                                                                    count].shape))
                    self.model = self.check_WB_limits(count, self.model)
                    if count % 2 == 1:
                        self.model.trainable_variables[count].assign(
                            np.zeros(self.model.trainable_variables[count].shape))

                # Update the learning rate
                self.learning_rate = max(self.learning_rate * 0.9995, 0.0001)
                self.learning_rate_cascaded = max(self.learning_rate_cascaded * 0.9995, 0.0001)
            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            # incremental_model.identify_incremental_model_LS(self.xt, ut_after)
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)
        else:
            Ec_actor_before = 0.5 * np.square(Jt1)
            # print("ACTOR LOSS xt1 before= ", Ec_actor_before)

            # Train the actor
            Jt1 = Jt1.flatten()[0]
            chain_rule = Jt1 * np.matmul(np.reshape(G[self.indices_tracking_states[0], :], [-1, 1]).T, dJt1_dxt1)

            chain_rule = chain_rule.flatten()[0]
            if self.time_step > self.start_training:
                for count in range(len(self.dut_dWb)):
                    gradient = chain_rule * self.dut_dWb[count]
                    self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * gradient,
                                                                                self.model.trainable_variables[
                                                                                    count].shape))
                    # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                    self.model = self.check_WB_limits(count, self.model)
                    if count % 2 == 1:
                        self.model.trainable_variables[count].assign(
                            np.zeros(self.model.trainable_variables[count].shape))
                # Update the learning rate
                self.learning_rate = max(self.learning_rate * 0.995, 0.001)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)
            # print("ACTOR LOSS xt1 after= ", Ec_actor_after)

    def compute_Adam_update(self, count, gradient, model, learning_rate):
        """
        Computes the Adam update and applies it to the weight updates.
        :param count: index of weight matrix being analysed
        :param gradient: computed gradient for each of the weights
        :param model: NN of the weights that are being updated
        :param learning_rate: the learning rate of the model being analysed
        :return: model --> return the updated model
                learning_rate --> return the updated learning rate
        """
        momentum = self.beta_momentum * self.momentum_dict[count] + (1 - self.beta_momentum) * gradient
        self.momentum_dict[count] = momentum
        momentum_corrected = momentum / (1 - np.power(self.beta_momentum, self.time_step + 1))

        rmsprop = self.beta_rmsprop * self.rmsprop_dict[count] + \
                  (1 - self.beta_rmsprop) * np.multiply(gradient, gradient)
        self.rmsprop_dict[count] = rmsprop
        rmsprop_corrected = rmsprop / (1 - np.power(self.beta_rmsprop, self.time_step + 1))

        update = momentum_corrected / (np.sqrt(rmsprop_corrected) + self.epsilon)
        model.trainable_variables[count].assign_sub(np.reshape(learning_rate * update,
                                                               model.trainable_variables[count].shape))
        # Implement WB_limits: the weights and biases can not have values whose absolute value
        # exceeds WB_limits
        model = self.check_WB_limits(count, model)
        if count % 2 == 1:
            model.trainable_variables[count].assign(np.zeros(model.trainable_variables[count].shape))

        if count == len(model.trainable_variables) - 1:
            learning_rate = max(learning_rate * 0.9995, 0.0001)

        return model, learning_rate

    def check_WB_limits(self, count, model):
        """
        Check whether any of the weights and biases exceed the limit imposed (WB_limits) and saturate the values
        :param count: index within the model.trainable_variables being analysed
        :return:
        """
        WB_variable = model.trainable_variables[count].numpy()
        WB_variable[WB_variable > self.WB_limits] = self.WB_limits
        WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
        model.trainable_variables[count].assign(WB_variable)
        return model

    def compute_persistent_excitation(self, *args):
        """
        Computation of the persistent excitation at each time step. Formula obtained from Pedro's thesis
        :return: e0 --> PE deviation
        """
        if len(args) == 1:
            t = args[0] + 1
        elif len(args) == 0:
            t = self.time_step + 1

        e0_1 = 0
        e0_2 = 0
        if self.type_PE == 'sinusoidal' or self.type_PE == 'combined':
            e0_1 = np.sin(t) * np.cos(2 * t) * (np.sin(3 * t + np.pi / 4) + np.cos(4 * t - np.pi / 3)) * 1e-2

        if self.type_PE == '3211' or self.type_PE == 'combined':
            if t < 3 * self.pulse_length_3211 / 7:
                e0_2 = 0.5 * self.amplitude_3211
            elif t < 5 * self.pulse_length_3211 / 7:
                e0_2 = -0.5 * self.amplitude_3211
            elif t < 6 * self.pulse_length_3211 / 7:
                e0_2 = 0.8 * self.amplitude_3211
            elif t < self.pulse_length_3211:
                e0_2 = -self.amplitude_3211

        e0 = e0_1 + e0_2

        return e0

    def update_actor_attributes(self):
        """
        The attributes that change with every time step are updated
        :return:
        """
        self.time_step += 1
        self.dut_dWb_1 = self.dut_dWb

        for counter in range(len(self.layers)):
            self.store_weights['W' + str(counter + 1)][:, self.time_step] = self.model.trainable_variables[
                counter * 2].numpy().flatten()

        if self.cascaded_actor:
            for counter in range(len(self.layers)):
                self.store_weights_q['W' + str(counter + 1)][:, self.time_step] = self.model_q.trainable_variables[
                    counter * 2].numpy().flatten()

    def evaluate_actor(self, *args):
        """
        Evaluation of the actor NN given an input or attributes stored in the object
        :param args: the real and reference states could be provided as input for the evaluation, or not if already stored
        :return: ut --> input to the system and the incremental model
        """
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
            tracked_states = np.reshape(xt[self.indices_tracking_states[0], :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = tf.constant(np.array([(xt_error)]).astype('float32'))

            q_ref_0 = self.model(nn_input)
            if self.activations[-1] == 'sigmoid':
                q_ref = max(min((2 * self.maximum_q_rate * q_ref_0.numpy()) - self.maximum_q_rate,
                                np.reshape(self.maximum_q_rate, q_ref_0.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref_0.numpy().shape))
            elif self.activations[-1] == 'tanh':
                q_ref = max(min((self.maximum_q_rate * q_ref_0.numpy()),
                                np.reshape(self.maximum_q_rate, q_ref_0.numpy().shape)),
                            np.reshape(-self.maximum_q_rate, q_ref_0.numpy().shape))

            tracked_states = np.reshape(xt[self.indices_tracking_states[1], :], [-1, 1])
            xt_error_q = np.reshape(tracked_states - np.reshape(q_ref, tracked_states.shape), [-1, 1])
            nn_input_q = tf.constant(np.array([(xt_error_q)]).astype('float32'))

            ut = self.model_q(nn_input_q).numpy()

        else:
            tracked_states = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
            xt_error = np.reshape(tracked_states - xt_ref, [-1, 1])
            nn_input = tf.constant(np.array([(xt_error)]).astype('float32'))

            ut = self.model(nn_input).numpy()

        if len(args) == 1:
            e0 = self.compute_persistent_excitation(time_step)
        else:
            e0 = self.compute_persistent_excitation()

        if self.activations[-1] == 'sigmoid':
            ut = max(min((2 * self.maximum_input * ut) - self.maximum_input + e0,
                         np.reshape(self.maximum_input, ut.shape)),
                     np.reshape(-self.maximum_input, ut.shape))
        elif self.activations[-1] == 'tanh':
            ut = max(min(((self.maximum_input + 10) * ut) + e0,
                         np.reshape(self.maximum_input, ut.shape)),
                     np.reshape(-self.maximum_input, ut.shape))
        return ut
    def restart_time_step(self):
        self.time_step = 0

    def restart_actor(self):
        """
        Restart the actor attributes
        :return:
        """
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

        # Restart momentum and rmsprop
        for count in range(len(self.model.trainable_variables)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0
