from abc import ABC, abstractmethod
from typing import List
from PIL import Image

from ..entities.classification_result import ClassificationResult
from ..enums.trash_category import TrashCategory


class IClassifier(ABC):
    """Abstract interface for all AI classifiers"""

    @abstractmethod
    def classify(self, image: Image.Image) -> ClassificationResult:
        """Classify an image and return result"""
        pass

    @abstractmethod
    def get_supported_classes(self) -> List[TrashCategory]:
        """Return list of classes this model can predict"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier"""
        pass

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Minimum confidence to accept prediction"""
        pass
