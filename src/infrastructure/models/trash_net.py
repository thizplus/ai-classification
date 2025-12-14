from typing import List
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

from src.domain.interfaces.classifier import IClassifier
from src.domain.entities.classification_result import ClassificationResult
from src.domain.enums.trash_category import TrashCategory


class TrashNetClassifier(IClassifier):
    """
    Trash-Net Classifier using HuggingFace model: prithivMLmods/Trash-Net

    Categories (6 classes):
    - cardboard (0)
    - glass (1)
    - metal (2)
    - paper (3)
    - plastic (4)
    - trash (5)
    """

    # Mapping from model index to TrashCategory enum
    INDEX_TO_CATEGORY = [
        TrashCategory.CARDBOARD,  # 0
        TrashCategory.GLASS,      # 1
        TrashCategory.METAL,      # 2
        TrashCategory.PAPER,      # 3
        TrashCategory.PLASTIC,    # 4
        TrashCategory.TRASH,      # 5
    ]

    MODEL_NAME = "prithivMLmods/Trash-Net"

    def __init__(self, threshold: float = 0.85, device: str = None):
        """
        Initialize TrashNet classifier

        Args:
            threshold: Minimum confidence threshold (default: 0.85)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self._threshold = threshold

        # Auto-detect device
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        print(f"[TrashNet] Loading model: {self.MODEL_NAME}")
        print(f"[TrashNet] Using device: {self._device}")

        # Load model and processor
        self._model = SiglipForImageClassification.from_pretrained(self.MODEL_NAME)
        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)

        # Move model to device
        self._model.to(self._device)
        self._model.eval()

        print(f"[TrashNet] Model loaded successfully!")

    def classify(self, image: Image.Image) -> ClassificationResult:
        """
        Classify an image

        Args:
            image: PIL Image (RGB)

        Returns:
            ClassificationResult with category and confidence
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess for model
        inputs = self._processor(images=image, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get prediction
        idx = probs.argmax().item()
        confidence = probs[0][idx].item()

        # Map to category enum
        category = self.INDEX_TO_CATEGORY[idx]

        return ClassificationResult(
            category=category,
            confidence=confidence,
            model_used=self.get_model_name()
        )

    def get_supported_classes(self) -> List[TrashCategory]:
        """Return list of supported categories"""
        return list(TrashCategory)

    def get_model_name(self) -> str:
        """Return model identifier"""
        return "trash_net_v1_hf"

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold"""
        return self._threshold
