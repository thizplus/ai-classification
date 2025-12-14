"""
IDetector Interface - Abstract interface for object detection (L0)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: int  # Top-left x
    y1: int  # Top-left y
    x2: int  # Bottom-right x
    y2: int  # Bottom-right y

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple for PIL crop"""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectionResult:
    """Result from object detection"""
    found: bool                          # Whether object was found
    bbox: Optional[BoundingBox] = None   # Bounding box if found
    confidence: float = 0.0              # Detection confidence
    label: str = ""                      # Detected object label (e.g., "bottle", "can")

    # For multiple objects (future use)
    all_detections: List['DetectionResult'] = None

    def __post_init__(self):
        if self.all_detections is None:
            self.all_detections = []


class IDetector(ABC):
    """Abstract interface for object detection (L0)"""

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Detect objects in image

        Args:
            image: PIL Image to detect objects in

        Returns:
            DetectionResult with bounding box if object found
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier"""
        pass

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Minimum confidence to accept detection"""
        pass

    @property
    @abstractmethod
    def supported_classes(self) -> List[str]:
        """List of classes this detector can detect"""
        pass
