from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image


class IImageProcessor(ABC):
    """Abstract interface for image processing"""

    @abstractmethod
    def load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load image from bytes"""
        pass

    @abstractmethod
    def load_from_url(self, url: str) -> Image.Image:
        """Load image from URL"""
        pass

    @abstractmethod
    def resize(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size"""
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Full preprocessing pipeline for model input"""
        pass
