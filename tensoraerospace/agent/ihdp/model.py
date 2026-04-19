"""IHDP agent wrapper.

This module defines the high-level IHDPAgent class that composes Actor, Critic,
and IncrementalModel components.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from .Actor import Actor
from .Critic import Critic
from .Incremental_model import IncrementalModel


class IHDPAgent(object):
    """IHDP Control Agent.

    Args:
        actor_settings (dict): Actor settings.
        critic_settings (dict): Critic settings.
        incremental_settings (dict): Incremental model settings.
        tracking_states: Tracked states.
        selected_states: Selected states.
        selected_input: Selected input signals.
        number_time_steps: Number of time steps.
        indices_tracking_states: Index of tracked states.
    """

    def __init__(
        self,
        actor_settings: dict,
        critic_settings: dict,
        incremental_settings: dict,
        tracking_states: list[str],
        selected_states: list[str],
        selected_input: list[str],
        number_time_steps: int,
        indices_tracking_states: list[int],
    ) -> None:
        """Compose IHDP agent components.

        Args:
            actor_settings: Configuration for Actor.
            critic_settings: Configuration for Critic.
            incremental_settings: Configuration for IncrementalModel.
            tracking_states: Tracked state names.
            selected_states: State variable names.
            selected_input: Control input names.
            number_time_steps: Episode length.
            indices_tracking_states: Indices of tracked states.
        """
        actor_keys = [
            "start_training",
            "layers",
            "activations",
            "learning_rate",
            "learning_rate_exponent_limit",
            "type_PE",
            "amplitude_3211",
            "pulse_length_3211",
            "maximum_input",
            "maximum_q_rate",
            "WB_limits",
            "NN_initial",
            "cascade_actor",
            "learning_rate_cascaded",
        ]
        critic_keys = [
            "Q_weights",
            "start_training",
            "gamma",
            "learning_rate",
            "learning_rate_exponent_limit",
            "layers",
            "activations",
            "indices_tracking_states",
            "WB_limits",
            "NN_initial",
        ]
        incremental_keys = [
            "number_time_steps",
            "dt",
            "input_magnitude_limits",
            "input_rate_limits",
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

        self.tracking_states = tracking_states
        self.selected_states = selected_states
        self.selected_input = selected_input
        self.number_time_steps = number_time_steps
        self.indices_tracking_states = indices_tracking_states
        # Keep the original settings dicts so ``save()`` can replay them.
        self.actor_settings = dict(actor_settings)
        self.critic_settings = dict(critic_settings)
        self.incremental_settings = dict(incremental_settings)

        self.actor = Actor(
            selected_input,
            selected_states,
            tracking_states,
            indices_tracking_states,
            number_time_steps,
            actor_settings["start_training"],
            actor_settings["layers"],
            actor_settings["activations"],
            actor_settings["learning_rate"],
            actor_settings["learning_rate_cascaded"],
            actor_settings["learning_rate_exponent_limit"],
            actor_settings["type_PE"],
            actor_settings["amplitude_3211"],
            actor_settings["pulse_length_3211"],
            actor_settings["WB_limits"],
            actor_settings["maximum_input"],
            actor_settings["maximum_q_rate"],
            actor_settings["cascade_actor"],
            actor_settings["NN_initial"],
        )
        self.actor.build_actor_model()

        self.critic = Critic(
            critic_settings["Q_weights"],
            selected_states,
            tracking_states,
            indices_tracking_states,
            number_time_steps,
            critic_settings["start_training"],
            critic_settings["gamma"],
            critic_settings["learning_rate"],
            critic_settings["learning_rate_exponent_limit"],
            critic_settings["layers"],
            critic_settings["activations"],
            critic_settings["WB_limits"],
            critic_settings["NN_initial"],
        )
        self.critic.build_critic_model()
        self.incremental_model = IncrementalModel(
            selected_states,
            selected_input,
            number_time_steps,
            incremental_settings["dt"],
            incremental_settings["input_magnitude_limits"],
            incremental_settings["input_rate_limits"],
        )

    def predict(
        self, xt: np.ndarray, reference_signals: np.ndarray, time_step: int
    ) -> np.ndarray:
        """Make prediction and get next control signals.

        Args:
            xt (_type_): Current state of the control object at step t.
            reference_signals (_type_): Reference control signal.
            time_step (_type_): Current time step.

        Returns:
            ut (_type_): Control signal at step t+1.
        """
        # Обработка входных состояний для совместимости с новой моделью F16
        xt = self._process_state_input(xt)

        # Если у нас больше состояний, чем отслеживаемых, извлекаем только нужные
        # Иначе используем все состояния (они уже отслеживаемые)
        if xt.shape[0] > len(self.indices_tracking_states):
            xt_tracked = xt[self.indices_tracking_states, :]
        else:
            xt_tracked = xt

        # Проверка размерности reference_signals
        if time_step >= reference_signals.shape[1]:
            raise ValueError(
                f"time_step {time_step} превышает размерность reference_signals {reference_signals.shape[1]}"
            )

        xt_ref = np.reshape(reference_signals[:, time_step], [-1, 1])
        ut = self.actor.run_actor_online(xt_tracked, xt_ref)

        G = self.incremental_model.identify_incremental_model_LS(xt, ut)
        xt1_est = self.incremental_model.evaluate_incremental_model()

        # Проверка для следующего временного шага
        if time_step + 1 >= reference_signals.shape[1]:
            # Используем последнее доступное значение reference_signal
            xt_ref1 = np.reshape(reference_signals[:, -1], [-1, 1])
        else:
            xt_ref1 = np.reshape(reference_signals[:, time_step + 1], [-1, 1])

        _ = self.critic.run_train_critic_online_alpha_decay(xt_tracked, xt_ref)
        Jt1, dJt1_dxt1 = self.critic.evaluate_critic(
            np.reshape(xt1_est, [-1, 1]), xt_ref1
        )
        self.actor.train_actor_online_alpha_decay(
            Jt1, dJt1_dxt1, G, self.incremental_model, self.critic, xt_ref1
        )

        self.incremental_model.update_incremental_model_attributes()
        self.critic.update_critic_attributes()
        self.actor.update_actor_attributes()
        return ut

    def _process_state_input(self, xt: np.ndarray | list) -> np.ndarray:
        """Process input states for compatibility with the new F16 model.

        Args:
            xt: Input state (can be in various formats)

        Returns:
            Processed state in the correct format.
        """
        # Конвертация в numpy array если необходимо
        if not isinstance(xt, np.ndarray):
            xt = np.array(xt)

        # Обработка различных форматов состояний
        if xt.ndim == 1:
            # Одномерный массив - преобразуем в столбец
            xt = xt.reshape([-1, 1])
        elif xt.ndim == 2:
            # Двумерный массив - проверяем ориентацию
            if xt.shape[1] > xt.shape[0] and xt.shape[0] == 1:
                # Строка - транспонируем в столбец
                xt = xt.T
            elif xt.shape[1] == 1:
                # Уже столбец - оставляем как есть
                pass
            else:
                # Неопределенный формат - берем первый столбец
                xt = xt[:, 0].reshape([-1, 1])
        else:
            # Многомерный массив - сплющиваем и делаем столбцом
            xt = xt.flatten().reshape([-1, 1])

        return xt

    # ------------------------------------------------------------------
    # Persistence — local save / load and Hugging Face Hub round-trip
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        """Build a JSON-serialisable config for :meth:`save`.

        Mirrors the structure used by other TensorAeroSpace agents:
        ``policy.name`` identifies the agent class, ``policy.params``
        captures everything the constructor needs.
        """
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return {
            "policy": {
                "name": agent_name,
                "params": {
                    "actor_settings": _json_clean(self.actor_settings),
                    "critic_settings": _json_clean(self.critic_settings),
                    "incremental_settings": _json_clean(self.incremental_settings),
                    "tracking_states": list(self.tracking_states),
                    "selected_states": list(self.selected_states),
                    "selected_input": list(self.selected_input),
                    "number_time_steps": int(self.number_time_steps),
                    "indices_tracking_states": list(self.indices_tracking_states),
                },
            },
        }

    def save(
        self,
        path: Union[str, Path, None] = None,
    ) -> str:
        """Write the agent to a directory.

        Files produced:
            * ``config.json`` — constructor kwargs (settings dicts +
              tracking/state metadata).
            * ``actor.pth`` / ``critic.pth`` — ``state_dict`` of the
              actor and critic inner ``torch.nn`` networks.

        Args:
            path: Base directory (``None`` → CWD).

        Returns:
            Absolute path to the created run directory.
        """
        base = Path.cwd() if path is None else Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = base / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        torch.save(self.actor.model.state_dict(), run_dir / "actor.pth")
        torch.save(self.critic.model.state_dict(), run_dir / "critic.pth")
        return str(run_dir)

    @classmethod
    def _load_from_dir(cls, folder: Union[str, Path]) -> "IHDPAgent":
        """Reconstruct an agent from a :meth:`save` directory."""
        folder_p = Path(folder)
        config_path = folder_p / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {str(folder_p)!r}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        policy = cfg.get("policy", {})
        params = policy.get("params", {})

        agent = cls(
            actor_settings=params["actor_settings"],
            critic_settings=params["critic_settings"],
            incremental_settings=params["incremental_settings"],
            tracking_states=params["tracking_states"],
            selected_states=params["selected_states"],
            selected_input=params["selected_input"],
            number_time_steps=params["number_time_steps"],
            indices_tracking_states=params["indices_tracking_states"],
        )

        actor_path = folder_p / "actor.pth"
        critic_path = folder_p / "critic.pth"
        if actor_path.exists():
            agent.actor.model.load_state_dict(
                torch.load(actor_path, map_location="cpu", weights_only=False)
            )
        if critic_path.exists():
            agent.critic.model.load_state_dict(
                torch.load(critic_path, map_location="cpu", weights_only=False)
            )
        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "IHDPAgent":
        """Load an agent from a local directory or Hugging Face Hub.

        Args:
            repo_name: Local folder path, or ``namespace/repo_name`` on
                the Hugging Face Hub.
            access_token: Hub access token for private repos.
            version: Hub revision / branch / tag.

        Returns:
            IHDPAgent: Reconstructed agent.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(
                f"Local directory not found: '{repo_name}'." " Please check the path."
            )

        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls._load_from_dir(folder_path)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Upload a :meth:`save` directory to the Hugging Face Hub.

        Args:
            repo_name: Target repository id, e.g. ``"me/my-ihdp"``.
            folder_path: Local folder produced by :meth:`save`.
            access_token: Hub access token.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )


def _json_clean(obj: Any) -> Any:
    """Recursively coerce tuples/numpy-scalars into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
