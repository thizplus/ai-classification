# Smart Trash AI Classification - Clean Architecture

## Overview

ระบบคัดแยกขยะอัจฉริยะ ออกแบบด้วย **Clean Architecture** เพื่อให้:
- รองรับ AI Models หลายตัว (Trash-Net, PlasticNet, MetalNet, etc.)
- เปลี่ยน/เพิ่ม Model ได้ง่าย โดยไม่ต้องแก้ code หลัก
- ทดสอบได้ง่าย (Unit Test แยกแต่ละ Layer)
- แยก concerns ชัดเจน

---

## Classification Flow (3-Level System)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LEVEL 0 (Object Detection - YOLO)                   │
│                                                                         │
│     ภาพขยะ ───────▶ ┌──────────────────────────┐                        │
│     (จาก R2)        │      YOLOv8 Detector     │                        │
│                     │   Object Detection       │                        │
│                     └────────────┬─────────────┘                        │
│                                  │                                      │
│              ┌───────────────────┼───────────────────┐                  │
│              ▼                   ▼                   ▼                  │
│         ไม่พบวัตถุ         พบ 1+ วัตถุ           Confidence ต่ำ        │
│              │           (+ Bounding Box)            │                  │
│              ▼                   │                   ▼                  │
│          REJECT              CROP ROI            REJECT                 │
│       "ไม่พบขยะ"                 │            "ไม่ชัดเจน"               │
│                                  ▼                                      │
└──────────────────────────────────┼──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           LEVEL 1 (Gatekeeper)                          │
│                                                                         │
│                    ┌──────────────────────────┐                         │
│    Cropped ROI ───▶│       Trash-Net          │                         │
│                    │   prithivMLmods/Trash-Net│                         │
│                    │      6 categories        │                         │
│                    └────────────┬─────────────┘                         │
│                                 │                                       │
│         ┌──────────┬────────────┼────────────┬──────────┬─────────┐     │
│         ▼          ▼            ▼            ▼          ▼         ▼     │
│    cardboard    glass        metal        paper     plastic    trash   │
│         │          │            │            │          │         │     │
└─────────┼──────────┼────────────┼────────────┼──────────┼─────────┼─────┘
          │          │            │            │          │         │
          ▼          ▼            ▼            ▼          ▼         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           LEVEL 2 (Specialists)                         │
│                                                                         │
│      ช่อง 1     GlassNet     MetalNet      ช่อง 4   PlasticNet   REJECT │
│     (direct)    (Future)     (Future)     (direct)     │               │
│                Clear/Brown  Aluminum/Steel         ┌───┴───┐           │
│                                                    ▼       ▼           │
│                                                  PET    HDPE           │
│                                                  PP     PE             │
└─────────────────────────────────────────────────────────────────────────┘
```

### L0 Benefits:
| ประโยชน์ | รายละเอียด |
|----------|------------|
| **กรองภาพว่าง** | ไม่มีวัตถุ → ไม่ต้อง classify |
| **ROI Crop** | ตัดพื้นหลัง → Input ที่ดีขึ้นสำหรับ L1 |
| **Reject ได้** | Confidence ต่ำ → แจ้งเตือน |
| **Bounding Box** | รู้ตำแหน่งวัตถุ → crop ได้แม่นยำ |

### Phase Implementation:

| Phase | Level | Model | Status | หน้าที่ |
|-------|-------|-------|--------|---------|
| **Phase 1.0** | L0 | YOLOv8 | **เพิ่มแล้ว** | Object Detection + ROI Crop |
| **Phase 1.1** | L1 | Trash-Net (HuggingFace) | **ทำแล้ว** | แยก 6 ประเภทหลัก |
| Phase 2 | L2 | PlasticNet | ถัดไป | แยกพลาสติก PET/HDPE/PP/PE |
| Phase 3 | L2 | MetalNet | อนาคต | แยก Aluminum/Steel |
| Phase 3 | L2 | GlassNet | อนาคต | แยก Clear/Brown |

---

## System Integration Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  ESP32-CAM  │      │  Cloudflare │      │   GoFiber   │      │   Python    │
│             │      │     R2      │      │   Backend   │      │  AI Service │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │                    │
       │  1. ถ่ายภาพ        │                    │                    │
       │  2. Upload ────────▶│                    │                    │
       │                    │                    │                    │
       │  3. Image URL ◀────│                    │                    │
       │                    │                    │                    │
       │  4. Send URL + metadata ────────────────▶│                    │
       │                    │                    │                    │
       │                    │                    │  5. Call AI ───────▶│
       │                    │                    │     (image_url)     │
       │                    │                    │                    │
       │                    │  6. Fetch image ◀──│────────────────────│
       │                    │─────────────────────────────────────────▶│
       │                    │                    │                    │
       │                    │                    │                    │  7. Trash-Net
       │                    │                    │                    │     Inference
       │                    │                    │                    │
       │                    │                    │                    │  8. If plastic:
       │                    │                    │                    │     → PlasticNet
       │                    │                    │                    │
       │                    │                    │  9. Result ◀───────│
       │                    │                    │                    │
       │  10. Response (bin, message) ◀──────────│                    │
       │                    │                    │                    │
       │  11. แสดงผล LED/จอ │                    │                    │
       ▼                    │                    │                    │
```

### Communication:
- **ESP32 → R2**: HTTP POST (multipart/form-data)
- **ESP32 → GoFiber**: HTTP POST (JSON with image URL)
- **GoFiber → Python AI**: HTTP POST หรือ gRPC
- **Response**: JSON format

---

## Project Structure

```
ai-classification/
├── src/
│   ├── domain/                    # Layer 1: Business Logic (Inner)
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── trash_item.py      # TrashItem entity
│   │   │   └── classification_result.py
│   │   ├── enums/
│   │   │   ├── __init__.py
│   │   │   ├── trash_category.py  # plastic, metal, glass, paper, trash
│   │   │   └── plastic_type.py    # PET, HDPE, PP, PE
│   │   └── interfaces/
│   │       ├── __init__.py
│   │       ├── classifier.py      # Abstract IClassifier
│   │       └── image_processor.py # Abstract IImageProcessor
│   │
│   ├── application/               # Layer 2: Use Cases
│   │   ├── __init__.py
│   │   ├── classify_trash.py      # Main classification use case
│   │   ├── routing_service.py     # Route to specialized model
│   │   └── dto/
│   │       ├── __init__.py
│   │       ├── classification_request.py
│   │       └── classification_response.py
│   │
│   ├── infrastructure/            # Layer 3: External Implementations
│   │   ├── __init__.py
│   │   ├── models/                # AI Model Implementations
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py      # BaseModelAdapter
│   │   │   ├── trash_net.py       # TrashNetClassifier
│   │   │   ├── plastic_net.py     # PlasticNetClassifier
│   │   │   ├── metal_net.py       # MetalNetClassifier
│   │   │   └── glass_net.py       # GlassNetClassifier
│   │   ├── image/
│   │   │   ├── __init__.py
│   │   │   ├── roi_cropper.py     # ROI Cropping
│   │   │   └── preprocessor.py    # Resize, normalize
│   │   └── config/
│   │       ├── __init__.py
│   │       └── model_config.py    # Model paths, thresholds
│   │
│   ├── presentation/              # Layer 4: API/Interface (Outer)
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py          # FastAPI routes
│   │   │   └── schemas.py         # Pydantic schemas
│   │   └── cli/
│   │       └── __init__.py
│   │
│   └── container.py               # Dependency Injection Container
│
├── models/                        # Pretrained model files (.h5, .tflite, .onnx)
│   ├── trash_net/
│   ├── plastic_net/
│   ├── metal_net/
│   └── glass_net/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── config/
│   └── settings.yaml              # App configuration
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│         (FastAPI, CLI, ESP32 Interface)                        │
├─────────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                            │
│         (Use Cases, DTOs, Routing Logic)                       │
├─────────────────────────────────────────────────────────────────┤
│                      DOMAIN LAYER                               │
│         (Entities, Interfaces, Business Rules)                 │
├─────────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE LAYER                           │
│         (AI Models, Image Processing, Config)                  │
└─────────────────────────────────────────────────────────────────┘

Dependency Rule: ลูกศรชี้เข้าด้านใน
- Presentation → Application → Domain ← Infrastructure
```

---

## Core Interfaces (Domain Layer)

### 1. IClassifier Interface

```python
# src/domain/interfaces/classifier.py
from abc import ABC, abstractmethod
from typing import List
from domain.entities.classification_result import ClassificationResult

class IClassifier(ABC):
    """Abstract interface สำหรับทุก AI Model"""

    @abstractmethod
    def classify(self, image: bytes) -> ClassificationResult:
        """Classify single image"""
        pass

    @abstractmethod
    def get_supported_classes(self) -> List[str]:
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
```

### 2. IImageProcessor Interface

```python
# src/domain/interfaces/image_processor.py
from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image

class IImageProcessor(ABC):
    """Abstract interface สำหรับ Image Processing"""

    @abstractmethod
    def load(self, image_bytes: bytes) -> Image.Image:
        """Load bytes to PIL Image"""
        pass

    @abstractmethod
    def crop_roi(self, image: Image.Image, roi: Tuple[int, int, int, int]) -> Image.Image:
        """Crop Region of Interest (x, y, width, height)"""
        pass

    @abstractmethod
    def resize(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size (width, height)"""
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Full preprocessing pipeline for model input

        Steps:
        1. Resize to model input size (e.g., 224x224)
        2. Normalize if needed

        Returns: PIL.Image.Image (model adapter จะแปลงเป็น tensor เอง)
        """
        pass
```

**Design Decision:**
- ใช้ `PIL.Image.Image` เป็น intermediate type
- Model adapter แปลงเป็น `torch.Tensor` หรือ `np.ndarray` เอง
- ทำให้ processor ไม่ผูกกับ framework ใดๆ

---

## Entities (Domain Layer)

### ClassificationResult

```python
# src/domain/entities/classification_result.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Union
from domain.enums.trash_category import TrashCategory, PlasticType, MetalType, GlassType

# Type alias for all category enums
CategoryType = Union[TrashCategory, PlasticType, MetalType, GlassType]

@dataclass
class ClassificationResult:
    category: CategoryType     # Enum เสมอ (ไม่ใช่ str)
    confidence: float          # 0.0 - 1.0
    model_used: str            # e.g., "trash_net", "plastic_net"
    sub_result: Optional['ClassificationResult'] = None  # L2 result
    metadata: Dict = field(default_factory=dict)

    def is_confident(self, threshold: float) -> bool:
        """Check confidence against given threshold (จาก classifier)"""
        return self.confidence >= threshold

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,  # Enum → string for JSON
            "confidence": self.confidence,
            "model_used": self.model_used,
            "sub_result": self.sub_result.to_dict() if self.sub_result else None
        }
```

### Category Enums

```python
# src/domain/enums/trash_category.py
from enum import Enum

class TrashCategory(Enum):
    """L1 Categories - จาก Trash-Net (6 classes)"""
    CARDBOARD = "cardboard"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    TRASH = "trash"  # Reject / Non-recyclable

class PlasticType(Enum):
    """L2 Plastic Types - จาก PlasticNet"""
    PET = "PET"      # ขวดน้ำ, ขวดใส
    HDPE = "HDPE"    # ขวดนม, แกลลอน, ขวดทึบ
    PP = "PP"        # กล่องอาหาร, ฝาขวด
    PE = "PE"        # ถุงพลาสติก

class MetalType(Enum):
    """L2 Metal Types - จาก MetalNet (Future)"""
    ALUMINUM = "aluminum"  # กระป๋องน้ำอัดลม
    STEEL = "steel"        # กระป๋องอาหาร

class GlassType(Enum):
    """L2 Glass Types - จาก GlassNet (Future)"""
    CLEAR = "clear"   # ขวดใส
    BROWN = "brown"   # ขวดสีน้ำตาล
    GREEN = "green"   # ขวดสีเขียว
```

---

## Application Layer (Use Cases)

### Routing Service

```python
# src/application/routing_service.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from domain.interfaces.classifier import IClassifier
from domain.enums.trash_category import TrashCategory

@dataclass
class RegisteredClassifier:
    """Wrapper for classifier with priority"""
    classifier: IClassifier
    priority: int = 0  # Higher = preferred
    enabled: bool = True

class RoutingService:
    """Route L1 result to specialized L2 model(s)

    Future-proof: รองรับหลาย model ต่อ category
    - PlasticNet_v1 (priority=1)
    - PlasticNet_v2 (priority=2) ← ใช้ตัวนี้
    """

    def __init__(self):
        # รองรับหลาย model ต่อ category
        self._routes: Dict[TrashCategory, List[RegisteredClassifier]] = {}

    def register(
        self,
        category: TrashCategory,
        classifier: IClassifier,
        priority: int = 0
    ):
        """Register L2 classifier for a category"""
        if category not in self._routes:
            self._routes[category] = []

        self._routes[category].append(
            RegisteredClassifier(classifier=classifier, priority=priority)
        )
        # Sort by priority (highest first)
        self._routes[category].sort(key=lambda x: x.priority, reverse=True)

    def get_classifier(self, category: TrashCategory) -> Optional[IClassifier]:
        """Get highest priority enabled classifier for category"""
        if category not in self._routes:
            return None

        for registered in self._routes[category]:
            if registered.enabled:
                return registered.classifier

        return None

    def get_all_classifiers(self, category: TrashCategory) -> List[IClassifier]:
        """Get all enabled classifiers for category (for ensemble/fallback)"""
        if category not in self._routes:
            return []

        return [r.classifier for r in self._routes[category] if r.enabled]

    def has_specialized_model(self, category: TrashCategory) -> bool:
        return category in self._routes and len(self._routes[category]) > 0
```

**Usage (Future):**
```python
# Register multiple versions
routing.register(TrashCategory.PLASTIC, plastic_net_v1, priority=1)
routing.register(TrashCategory.PLASTIC, plastic_net_v2, priority=2)  # ใช้ตัวนี้

# Get best model
classifier = routing.get_classifier(TrashCategory.PLASTIC)  # → v2

# Ensemble (future)
all_classifiers = routing.get_all_classifiers(TrashCategory.PLASTIC)
```

### Main Classification Use Case

```python
# src/application/classify_trash.py
from typing import Optional, Tuple
from domain.interfaces.classifier import IClassifier
from domain.interfaces.image_processor import IImageProcessor
from domain.entities.classification_result import ClassificationResult
from domain.enums.trash_category import TrashCategory
from application.routing_service import RoutingService

class ClassifyTrashUseCase:
    """Main use case: 2-level classification"""

    def __init__(
        self,
        l1_classifier: IClassifier,  # Trash-Net
        routing_service: RoutingService,
        image_processor: IImageProcessor
    ):
        self._l1 = l1_classifier
        self._router = routing_service
        self._processor = image_processor

    def execute(
        self,
        raw_image: bytes,
        roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    ) -> ClassificationResult:
        """
        Execute 2-level classification

        Args:
            raw_image: Raw image bytes from camera
            roi: Optional Region of Interest (x, y, width, height)
        """
        # Step 1: Load bytes → PIL Image
        image = self._processor.load(raw_image)

        # Step 2: ROI Crop (Business rule - ก่อน technical preprocess)
        if roi:
            image = self._processor.crop_roi(image, roi)

        # Step 3: Technical Preprocess (resize, normalize)
        processed = self._processor.preprocess(image)

        # Step 4: L1 Classification (Trash-Net)
        l1_result = self._l1.classify(processed)

        # Step 5: Route to L2 if L1 confident
        l2_classifier = self._router.get_classifier(l1_result.category)

        if l2_classifier and l1_result.is_confident(self._l1.confidence_threshold):
            # ✅ ใช้ภาพเดียวกัน (processed) - ไม่ crop/ถ่ายใหม่
            l2_result = l2_classifier.classify(processed)

            # Step 6: L2 reject logic - ถ้าไม่มั่นใจ ไม่ attach sub_result
            if l2_result.is_confident(l2_classifier.confidence_threshold):
                l1_result.sub_result = l2_result
            else:
                # L2 ไม่มั่นใจ → ใช้แค่ L1 result + บันทึก note
                l1_result.metadata["l2_attempted"] = True
                l1_result.metadata["l2_rejected_reason"] = "low_confidence"
                l1_result.metadata["l2_confidence"] = l2_result.confidence

        return l1_result
```

**สำคัญ:**
- `load()` = bytes → PIL.Image
- `crop_roi()` = Business rule (ตัดเฉพาะส่วนที่สนใจ)
- `preprocess()` = Technical step (resize 224x224, normalize)
- L2 ใช้ภาพเดียวกับ L1 (ไม่ crop/ถ่ายใหม่)
- **L2 reject logic**: ถ้า L2 confidence ต่ำ → ไม่ attach sub_result → ส่ง bin รวม

---

## Infrastructure Layer (Model Implementations)

### Base Model Adapter

```python
# src/infrastructure/models/base_model.py
from abc import ABC
from typing import List
from domain.interfaces.classifier import IClassifier
from domain.entities.classification_result import ClassificationResult

class BaseModelAdapter(IClassifier, ABC):
    """Base class for all model implementations"""

    def __init__(self, model_path: str, threshold: float = 0.85):
        self._model_path = model_path
        self._threshold = threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        """Override to load specific model format"""
        raise NotImplementedError

    @property
    def confidence_threshold(self) -> float:
        return self._threshold
```

### Trash-Net Implementation (HuggingFace)

```python
# src/infrastructure/models/trash_net.py
import numpy as np
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

from infrastructure.models.base_model import BaseModelAdapter
from domain.entities.classification_result import ClassificationResult
from domain.enums.trash_category import TrashCategory

class TrashNetClassifier(BaseModelAdapter):
    """L1 Classifier using prithivMLmods/Trash-Net from HuggingFace"""

    # Mapping: model index → Enum (ตาม HuggingFace model)
    INDEX_TO_CATEGORY = [
        TrashCategory.CARDBOARD,  # 0
        TrashCategory.GLASS,      # 1
        TrashCategory.METAL,      # 2
        TrashCategory.PAPER,      # 3
        TrashCategory.PLASTIC,    # 4
        TrashCategory.TRASH,      # 5
    ]

    def _load_model(self):
        """Load HuggingFace model"""
        model_name = "prithivMLmods/Trash-Net"
        self._model = SiglipForImageClassification.from_pretrained(model_name)
        self._processor = AutoImageProcessor.from_pretrained(model_name)

    def classify(self, image: Image.Image) -> ClassificationResult:
        """Classify image and return Enum category"""
        # Preprocess for model
        inputs = self._processor(images=image, return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get best prediction
        idx = probs.argmax().item()
        confidence = probs[0][idx].item()

        return ClassificationResult(
            category=self.INDEX_TO_CATEGORY[idx],  # ✅ Return Enum
            confidence=confidence,
            model_used=self.get_model_name()
        )

    def get_supported_classes(self) -> list:
        return list(TrashCategory)

    def get_model_name(self) -> str:
        return "trash_net_v1_hf"
```

### Plastic-Net Implementation (Future - Phase 2)

```python
# src/infrastructure/models/plastic_net.py
import numpy as np
from PIL import Image
import torch

from infrastructure.models.base_model import BaseModelAdapter
from domain.entities.classification_result import ClassificationResult
from domain.enums.trash_category import PlasticType

class PlasticNetClassifier(BaseModelAdapter):
    """L2 Classifier for Plastic - Fine-tuned MobileNet/EfficientNet"""

    # Mapping: model index → Enum
    INDEX_TO_TYPE = [
        PlasticType.PET,   # 0 - ขวดน้ำ, ขวดใส
        PlasticType.HDPE,  # 1 - ขวดนม, แกลลอน
        PlasticType.PP,    # 2 - กล่องอาหาร, ฝาขวด
        PlasticType.PE,    # 3 - ถุงพลาสติก
    ]

    def __init__(self, model_path: str, threshold: float = 0.80):
        """PlasticNet ใช้ threshold ต่ำกว่า Trash-Net เพราะ classes คล้ายกัน"""
        super().__init__(model_path, threshold)

    def _load_model(self):
        """Load fine-tuned model (TFLite/ONNX/PyTorch)"""
        # TODO: Load actual model when trained
        # self._model = torch.load(self._model_path)
        pass

    def classify(self, image: Image.Image) -> ClassificationResult:
        """Classify plastic type"""
        # TODO: Implement actual inference
        # Mock for now
        predictions = [0.92, 0.05, 0.02, 0.01]

        idx = np.argmax(predictions)
        confidence = predictions[idx]

        return ClassificationResult(
            category=self.INDEX_TO_TYPE[idx],  # ✅ Return Enum
            confidence=float(confidence),
            model_used=self.get_model_name()
        )

    def get_supported_classes(self) -> list:
        return list(PlasticType)

    def get_model_name(self) -> str:
        return "plastic_net_v1"
```

**Note:** PlasticNet ใช้ `threshold=0.80` (ต่ำกว่า Trash-Net 0.85) เพราะ PET/HDPE/PP มีลักษณะคล้ายกัน

---

## Adding New Model (Future)

เมื่อต้องการเพิ่ม Model ใหม่ ทำตามขั้นตอนนี้:

### 1. สร้าง Model Class ใหม่

```python
# src/infrastructure/models/paper_net.py
from infrastructure.models.base_model import BaseModelAdapter

class PaperNetClassifier(BaseModelAdapter):
    CLASSES = ["cardboard", "newspaper", "office_paper", "magazine"]

    def _load_model(self):
        # Load your model
        pass

    def classify(self, image):
        # Implement inference
        pass

    def get_supported_classes(self):
        return self.CLASSES

    def get_model_name(self):
        return "paper_net_v1"
```

### 2. Register ใน Container

```python
# src/container.py
from infrastructure.models.paper_net import PaperNetClassifier

# Register L2 model
paper_classifier = PaperNetClassifier("models/paper_net/model.tflite")
routing_service.register(TrashCategory.PAPER, paper_classifier)
```

**ไม่ต้องแก้ code อื่นเลย!** Clean Architecture ทำให้ extend ได้ง่าย

---

## API Response Format

```json
{
  "success": true,
  "result": {
    "category": "plastic",
    "confidence": 0.96,
    "model_used": "trash_net_v1",
    "sub_result": {
      "category": "PET",
      "confidence": 0.92,
      "model_used": "plastic_net_v1"
    }
  },
  "action": {
    "bin_number": 1,
    "bin_label": "PET Bottles",
    "message": "ทิ้งช่อง 1 - ขวด PET"
  }
}
```

---

## Implementation Phases

### Phase 1: Trash-Net MVP (ทำก่อน)
- [ ] Setup project structure ตาม Clean Architecture
- [ ] Create domain layer (IClassifier interface, entities)
- [ ] Implement `TrashNetClassifier` ใช้ HuggingFace model
- [ ] สร้าง inference endpoint (Python service)
- [ ] ต่อกับ GoFiber backend (รับ image URL จาก R2)
- [ ] ทดสอบ end-to-end: ESP32 → R2 → GoFiber → Python → Response

**Output Phase 1:**
```json
{
  "category": "plastic",
  "confidence": 0.96,
  "bin": 5,
  "message": "ทิ้งช่อง 5 - พลาสติก"
}
```

### Phase 2: PlasticNet (L2 สำหรับพลาสติก)
- [ ] เก็บ dataset พลาสติกจริง (PET, HDPE, PP, PE)
- [ ] Fine-tune MobileNet/EfficientNet
- [ ] Implement `PlasticNetClassifier`
- [ ] เพิ่ม routing logic: ถ้า L1 = plastic → เรียก PlasticNet

**Output Phase 2:**
```json
{
  "category": "plastic",
  "confidence": 0.96,
  "sub_category": "PET",
  "sub_confidence": 0.92,
  "bin": 5,
  "message": "ทิ้งช่อง 5 - ขวด PET"
}
```

### Phase 3: Additional L2 Models (อนาคต)
- [ ] MetalNet: Aluminum vs Steel
- [ ] GlassNet: Clear vs Brown
- [ ] PaperNet: Cardboard vs Office paper (ถ้าจำเป็น)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Web Framework | FastAPI |
| AI Framework | TensorFlow Lite / ONNX Runtime |
| Image Processing | OpenCV, Pillow |
| DI Container | dependency-injector |
| Config | pydantic-settings |
| Testing | pytest |

---

## Key Benefits of This Architecture

1. **Model Agnostic**: เปลี่ยนจาก TFLite เป็น ONNX ได้โดยแก้แค่ infrastructure layer
2. **Easy Testing**: Mock `IClassifier` interface เพื่อ test use case
3. **Scalable**: เพิ่ม model ใหม่โดยไม่ต้องแก้ code เดิม
4. **Maintainable**: แยก concerns ชัดเจน แก้ bug ง่าย
5. **Framework Independent**: เปลี่ยนจาก FastAPI เป็น Flask ได้ง่าย
