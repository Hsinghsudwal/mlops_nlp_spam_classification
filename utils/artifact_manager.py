from abc import ABC, abstractmethod

class ArtifactManager(ABC):
    """Abstract base class for managing artifacts."""

    @abstractmethod
    def save(self, artifact, subdir: str, name: str, pipeline_id: str = None) -> str:
        pass

    @abstractmethod
    def load(self, subdir: str, name: str):
        pass

    @abstractmethod
    def get_base_path(self) -> str:
        pass

    @abstractmethod
    def resolve_path(self, relative_path: str) -> str:
        pass
