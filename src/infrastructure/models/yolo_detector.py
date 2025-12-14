"""
YOLODetector - Object detection using YOLOv8 (L0)

Uses ultralytics YOLOv8 for detecting objects in images.
Filters for trash-related objects (bottle, cup, can, etc.)
"""
from typing import List
from PIL import Image

from src.domain.interfaces.detector import IDetector, DetectionResult, BoundingBox


class YOLODetector(IDetector):
    """L0 Object Detector using YOLOv8"""

    # COCO classes that are relevant for trash detection
    TRASH_CLASSES = {
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        76: "scissors",
        77: "teddy bear",
        # General objects that could be trash
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        73: "book",
        74: "clock",
        75: "vase",
    }

    def __init__(self, model_name: str = "yolov8n.pt", threshold: float = 0.3):
        """
        Initialize YOLO detector

        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)
            threshold: Minimum confidence for detection
        """
        self._model_name = model_name
        self._threshold = threshold
        self._model = None
        self._device = "cpu"
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            import torch

            print(f"[YOLO] Loading model: {self._model_name}")

            # Check for GPU
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"

            print(f"[YOLO] Using device: {self._device}")

            self._model = YOLO(self._model_name)
            print("[YOLO] Model loaded successfully!")

        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
            raise

    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Detect objects in image

        Returns the largest/most confident trash-related object found
        """
        if self._model is None:
            return DetectionResult(found=False)

        # Run inference
        results = self._model(image, verbose=False, device=self._device)

        if not results or len(results) == 0:
            return DetectionResult(found=False)

        result = results[0]

        # Find best trash-related detection
        best_detection = None
        best_area = 0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Skip if confidence too low
            if conf < self._threshold:
                continue

            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

            # Check if this is a trash-related class OR just take largest object
            label = self.TRASH_CLASSES.get(cls_id, result.names.get(cls_id, "unknown"))

            # Prefer larger objects (more likely to be the main subject)
            if bbox.area > best_area:
                best_area = bbox.area
                best_detection = DetectionResult(
                    found=True,
                    bbox=bbox,
                    confidence=conf,
                    label=label
                )

        if best_detection:
            print(f"[YOLO] Detected: {best_detection.label} ({best_detection.confidence:.1%})")
            return best_detection

        # No trash-related object found, but if ANY object detected, return it
        if len(result.boxes) > 0:
            box = result.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf >= self._threshold:
                return DetectionResult(
                    found=True,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    label=result.names.get(cls_id, "object")
                )

        return DetectionResult(found=False)

    def get_model_name(self) -> str:
        return f"yolov8_{self._model_name}"

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    @property
    def supported_classes(self) -> List[str]:
        return list(self.TRASH_CLASSES.values())
