"""NARX neural network model.

This module defines a NARX (Nonlinear AutoRegressive with eXogenous inputs)
neural network model used for time-series prediction in some agents.
"""

import datetime
import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class NARX(nn.Module):
    """NARX (Nonlinear AutoRegressive with eXogenous inputs) MLP model.

    This is a simple fully-connected NARX-style network for time-series
    prediction that uses the current input and the previous output.

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Hidden layer size.
        output_size (int): Output feature dimension.

    Attributes:
        hidden_size (int): Hidden layer size.
        input_layer (nn.Linear): Linear layer applied to concatenated input and
            last output.
        output_layer (nn.Linear): Output projection layer.
        criterion (nn.MSELoss): Mean squared error loss.
        optimizer (torch.optim.Adam): Adam optimizer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize NARX MLP.

        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden layer width.
            output_size: Output feature dimension.
        """
        super(NARX, self).__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + output_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input_tensor, last_output):
        """Compute one forward step.

        Args:
            input_tensor (Tensor): Current input tensor.
            last_output (Tensor): Previous output tensor.

        Returns:
            Tensor: Model output tensor.
        """
        combined = torch.cat((input_tensor, last_output), 0)
        hidden = torch.tanh(self.input_layer(combined))
        output = self.output_layer(hidden)
        return output

    def train(self, predcit_tensor, target_tensor):
        """Perform one gradient update step.

        Note:
            This method name shadows ``torch.nn.Module.train``.

        Args:
            predcit_tensor (Tensor): Predicted tensor.
            target_tensor (Tensor): Target tensor.

        Returns:
            float: Loss value after one update step.
        """
        loss = self.criterion(predcit_tensor, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_param_env(self) -> dict:
        """Return serializable configuration of the NARX model."""
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        model_name = f"{module_name}.{class_name}"

        # Reconstruct input_size from the layer dimensions.
        input_layer_in = self.input_layer.in_features
        output_size = self.output_layer.out_features
        input_size = input_layer_in - output_size

        model_params = {
            "input_size": input_size,
            "hidden_size": self.hidden_size,
            "output_size": output_size,
        }

        return {
            "model": {"name": model_name, "params": model_params},
        }

    def save(
        self,
        path: Union[str, Path, None] = None,
        save_gradients: bool = False,
    ) -> Path:
        """Save NARX model to the specified directory.

        Saves the network state dict and configuration.
        Optionally saves the optimizer state for resuming training.

        Args:
            path (str | Path | None): Base save directory.  If *None*, saves to
                the current working directory.
            save_gradients (bool): If True, also save optimizer state dict.

        Returns:
            Path: The directory where the model was saved.
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        save_dir = path / f"{date_str}_{self.__class__.__name__}"
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "config.json"
        model_path = save_dir / "model.pth"
        optimizer_path = save_dir / "optimizer.pth"

        # Save configuration
        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save network weights
        torch.save(self.state_dict(), model_path)

        # Optionally save optimizer state
        if save_gradients:
            torch.save(self.optimizer.state_dict(), optimizer_path)

        return save_dir

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        load_gradients: bool = False,
    ) -> "NARX":
        """Load a NARX model from a checkpoint directory.

        Args:
            path: Directory containing saved model files.
            load_gradients: If True, restore optimizer state.

        Returns:
            NARX: Reconstructed model.
        """
        path = Path(path)
        config_path = path / "config.json"
        model_path = path / "model.pth"
        optimizer_path = path / "optimizer.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_params = config["model"]["params"]

        new_model = cls(
            input_size=model_params["input_size"],
            hidden_size=model_params["hidden_size"],
            output_size=model_params["output_size"],
        )

        # Restore network weights
        new_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False)
        )

        # Optionally restore optimizer state
        if load_gradients and optimizer_path.exists():
            new_model.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=False)
            )

        return new_model

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "NARX":
        """Load pretrained model from a local directory or Hugging Face Hub.

        Args:
            repo_name: Path to a local folder **or** a Hugging Face repo id
                (e.g. ``"namespace/repo_name"``).
            access_token: Hugging Face access token for private repos.
            version: Revision / branch / tag on Hugging Face.
            load_gradients: Restore optimizer state for continued training.

        Returns:
            NARX: Initialized model.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.load(p, load_gradients=load_gradients)

        # If it looks like an explicit filesystem path, raise immediately.
        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(f"Local directory not found: '{repo_name}'.")

        # Fall back to Hugging Face Hub download.
        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls.load(folder_path, load_gradients=load_gradients)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Upload a saved model folder to Hugging Face Hub.

        Args:
            repo_name: Repository id on Hugging Face (e.g. ``"user/my-narx"``).
            folder_path: Local folder produced by :meth:`save`.
            access_token: Hugging Face access token.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )
