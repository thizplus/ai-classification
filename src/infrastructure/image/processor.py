from typing import Tuple
from io import BytesIO
from PIL import Image
import httpx

from src.domain.interfaces.image_processor import IImageProcessor


class ImageProcessor(IImageProcessor):
    """Image processor implementation"""

    DEFAULT_SIZE = (224, 224)
    TIMEOUT = 30.0  # seconds

    def __init__(self, target_size: Tuple[int, int] = None):
        """
        Initialize image processor

        Args:
            target_size: Target size for preprocessing (default: 224x224)
        """
        self._target_size = target_size or self.DEFAULT_SIZE

    def load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load image from bytes"""
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def load_from_url(self, url: str) -> Image.Image:
        """
        Load image from URL

        Args:
            url: Image URL

        Returns:
            PIL Image (RGB)

        Raises:
            httpx.HTTPError: If fetch fails
            Exception: If image cannot be loaded
        """
        print(f"[ImageProcessor] Fetching image from: {url[:80]}...")

        # Fetch image
        with httpx.Client(timeout=self.TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
            image_bytes = response.content

        print(f"[ImageProcessor] Downloaded {len(image_bytes)} bytes")

        # Load image
        image = self.load_from_bytes(image_bytes)
        print(f"[ImageProcessor] Image loaded: {image.size}, mode: {image.mode}")

        return image

    def resize(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size"""
        return image.resize(size, Image.Resampling.LANCZOS)

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Full preprocessing pipeline

        1. Convert to RGB if needed
        2. Resize to target size
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if needed
        if image.size != self._target_size:
            image = self.resize(image, self._target_size)

        return image
