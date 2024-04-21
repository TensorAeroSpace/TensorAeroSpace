from abc import ABC
from huggingface_hub import HfApi, snapshot_download


class BaseRLModel(ABC):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    def get_env():
        pass
    
    def train():
        pass
    
    def action_probability():
        pass
    
    def save():
        pass
    
    def load():
        pass
    
    def predict():
        pass
    
    def get_param_env():
        pass

    def publish_to_hub(self, repo_name, folder_path, access_token=None):
        api = HfApi()
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )

    def from_pretrained(self, repo_name, access_token=None, version=None):
        folder_path = snapshot_download(repo_id=repo_name, token=access_token, revision=version)
        return folder_path