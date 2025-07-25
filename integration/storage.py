import os
import io
import json
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.logger import logger
from utils.artifact_manager import ArtifactManager
from utils.config_manager import ConfigManager
import boto3
from botocore.exceptions import NoCredentialsError


class LocalArtifactStore(ArtifactManager):
    def __init__(self, config: ConfigManager):
        self.config = config
        self.base_path = self.config.get("artifact_path.path", "outputs")
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"LocalArtifactStore initialized at: {self.base_path}")

    def save(self, artifact, subdir: str, name: str, pipeline_id: str = None) -> str:
        path = os.path.join(self.base_path, subdir)
        os.makedirs(path, exist_ok=True)
        artifact_path = os.path.join(path, name)

        try:
            ext = name.split('.')[-1].lower()
            if ext == "pkl":
                with open(artifact_path, "wb") as f:
                    pickle.dump(artifact, f)
            elif ext == "csv" and isinstance(artifact, pd.DataFrame):
                artifact.to_csv(artifact_path, index=False)
            elif ext == "json":
                with open(artifact_path, "w") as f:
                    json.dump(artifact, f, indent=4)
            elif ext == "txt":
                with open(artifact_path, "w") as f:
                    f.write(str(artifact))
            elif ext == "png":
                if isinstance(artifact, plt.Figure):
                    artifact.savefig(artifact_path)
                elif isinstance(artifact, Image.Image):
                    artifact.save(artifact_path)
                elif isinstance(artifact, np.ndarray):
                    Image.fromarray(artifact).save(artifact_path)
                else:
                    raise ValueError("PNG format supports matplotlib, PIL, or numpy image arrays.")
            else:
                with open(artifact_path, "wb") as f:
                    pickle.dump(artifact, f)

            logger.info(f"Saved artifact: {artifact_path}")
            return artifact_path
        except Exception as e:
            logger.error(f"Failed to save artifact '{name}': {e}")
            raise

    def load(self, subdir: str, name: str):
        artifact_path = os.path.join(self.base_path, subdir, name)
        if not os.path.exists(artifact_path):
            logger.warning(f"Artifact not found: {artifact_path}")
            return None

        try:
            ext = name.split('.')[-1].lower()
            if ext == "pkl":
                with open(artifact_path, "rb") as f:
                    return pickle.load(f)
            elif ext == "csv":
                return pd.read_csv(artifact_path)
            elif ext == "json":
                with open(artifact_path, "r") as f:
                    return json.load(f)
            elif ext == "txt":
                with open(artifact_path, "r") as f:
                    return f.read()
            else:
                with open(artifact_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load artifact '{name}': {e}")
            raise

    def get_base_path(self) -> str:
        return self.base_path

    def resolve_path(self, relative_path: str) -> str:
        return os.path.join(self.base_path, relative_path)


class LocalStackArtifactStore(LocalArtifactStore):
    """Simulated S3 artifact store using LocalStack."""

    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.bucket_name = config.get("s3.bucket", "ml-artifacts")
        self.endpoint_url = config.get("s3.endpoint_url", "http://localhost:4566")
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            endpoint_url=self.endpoint_url,
            region_name=config.get("s3.region", "us-east-1"),
        )
        logger.info(f"LocalStackArtifactStore initialized: {self.bucket_name} @ {self.endpoint_url}")


    def save(self, artifact, subdir: str, name: str, pipeline_id: str = None) -> str:
        pass

    def load(self, subdir: str, name: str):
        pass
    
    def get_base_path(self) -> str:
        pass

    def resolve_path(self, relative_path: str) -> str:
        pass
        
class CloudArtifactStore(ArtifactManager):

    def save(self, artifact, subdir: str, name: str, pipeline_id: str = None) -> str:
        pass
    
    def load(self, subdir: str, name: str):
        pass

    def get_base_path(self) -> str:
        pass

    def resolve_path(self, relative_path: str) -> str:
        pass


class ArtifactStoreFactory:
    @staticmethod
    def create_store(config: ConfigManager) -> ArtifactManager:
        store_type = config.get("storage_type", "local").lower()
        if store_type == "local":
            return LocalArtifactStore(config)
        elif store_type == "localstack":
            return LocalStackArtifactStore(config)
        elif store_type == "cloud":
            return CloudArtifactStore(config)
        else:
            raise ValueError(f"Unsupported storage type: {store_type}")
