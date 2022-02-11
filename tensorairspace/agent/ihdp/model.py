from tensorairspace.agent.ihdp.Incremental_model import IncrementalModel
from tensorairspace.agent.ihdp.Actor import Actor
from tensorairspace.agent.ihdp.Critic import Critic
import numpy as np


class IHDPAgent(object):
    def __init__(self, actor_settings: dict, critic_settings: dict, incremental_settings: dict, tracking_states,
                 selected_states, selected_input, number_time_steps, indices_tracking_states):
        actor_keys = [
            "start_training",
            "layers", "activations", "learning_rate", "learning_rate_exponent_limit",
            "type_PE", "amplitude_3211", "pulse_length_3211", "maximum_input", "maximum_q_rate",
            "WB_limits", "NN_initial",
            "cascade_actor", "learning_rate_cascaded"
        ]
        critic_keys = [
            "Q_weights", "start_training", "gamma", "learning_rate", "learning_rate_exponent_limit",
            "layers", "activations", "indices_tracking_states", "WB_limits", "NN_initial",
        ]
        incremental_keys = [
            "number_time_steps", "dt", "input_magnitude_limits", "input_rate_limits"
        ]
        for key in actor_keys:
            if key not in actor_settings.keys():
                raise Exception(f"Key {key} not in actor settings")

        for key in critic_keys:
            if key not in critic_settings.keys():
                raise Exception(f"Key {key} not in critic settings")

        for key in incremental_keys:
            if key not in incremental_settings.keys():
                raise Exception(f"Key {key} not in incremental settings")

        self.actor = Actor(selected_input, selected_states, tracking_states, indices_tracking_states,
                           number_time_steps, actor_settings["start_training"],
                           actor_settings["layers"], actor_settings["activations"],
                           actor_settings["learning_rate"], actor_settings["learning_rate_cascaded"],
                           actor_settings["learning_rate_exponent_limit"],
                           actor_settings["type_PE"], actor_settings["amplitude_3211"],
                           actor_settings["pulse_length_3211"],
                           actor_settings["WB_limits"],
                           actor_settings["maximum_input"], actor_settings["maximum_q_rate"],
                           actor_settings["cascade_actor"], actor_settings["NN_initial"])
        self.actor.build_actor_model()

        self.critic = Critic(critic_settings["Q_weights"], selected_states, tracking_states,
                             indices_tracking_states,
                             number_time_steps,
                             critic_settings["start_training"], critic_settings["gamma"],
                             critic_settings["learning_rate"],
                             critic_settings["learning_rate_exponent_limit"],
                             critic_settings["layers"],
                             critic_settings["activations"], critic_settings["WB_limits"],
                             critic_settings["NN_initial"])
        self.critic.build_critic_model()
        self.incremental_model = IncrementalModel(selected_states, selected_input, number_time_steps,
                                                  incremental_settings["dt"],
                                                  incremental_settings["input_magnitude_limits"],
                                                  incremental_settings["input_rate_limits"])

    def predict(self, xt, reference_signals, time_step):
        xt_ref = np.reshape(reference_signals[:, time_step], [-1, 1])
        ut = self.actor.run_actor_online(xt, xt_ref)

        G = self.incremental_model.identify_incremental_model_LS(xt, ut)
        xt1_est = self.incremental_model.evaluate_incremental_model()

        xt_ref1 = np.reshape(reference_signals[:, time_step + 1], [-1, 1])

        _ = self.critic.run_train_critic_online_alpha_decay(xt, xt_ref)
        Jt1, dJt1_dxt1 = self.critic.evaluate_critic(np.reshape(xt1_est, [-1, 1]), xt_ref1)
        self.actor.train_actor_online_alpha_decay(Jt1, dJt1_dxt1, G, self.incremental_model, self.critic, xt_ref1)

        self.incremental_model.update_incremental_model_attributes()
        self.critic.update_critic_attributes()
        self.actor.update_actor_attributes()
        return ut
