from typing import Optional
import traceback
from src.domain.interfaces.classifier import IClassifier
from src.domain.interfaces.image_processor import IImageProcessor
from src.domain.interfaces.detector import IDetector
from src.domain.entities.classification_result import ClassificationResult


class ClassifyTrashUseCase:
    """
    Main use case for trash classification (3-Level System)

    Flow:
    1. Load image from URL
    2. L0: Object Detection (YOLO) - find object and crop ROI
    3. L1: Material Classification (Trash-Net)
    4. (Future) L2: Sub-type classification
    5. Return result
    """

    def __init__(
        self,
        classifier: IClassifier,
        image_processor: IImageProcessor,
        detector: Optional[IDetector] = None
    ):
        self._classifier = classifier
        self._processor = image_processor
        self._detector = detector

    def execute(self, image_url: str) -> ClassificationResult:
        """Execute classification with L0 → L1 flow"""
        print(f"[UseCase] Starting classification...", flush=True)

        # Step 1: Load image from URL
        try:
            print(f"[UseCase] Loading image from URL...", flush=True)
            image = self._processor.load_from_url(image_url)
            print(f"[UseCase] Image loaded: {image.size}", flush=True)
        except Exception as e:
            print(f"[UseCase] ERROR loading image: {e}", flush=True)
            traceback.print_exc()
            raise

        # Step 2: L0 - Object Detection (if detector available)
        # L0 แค่รายงานว่าเจอ/ไม่เจออะไร ไม่มีสิทธิ์หยุด flow
        # ส่งต่อให้ L1 เสมอ เพื่อให้เห็นปัญหาและ debug ได้
        detection = None
        if self._detector:
            try:
                print(f"[UseCase] Running L0 (YOLO) detection...", flush=True)
                detection = self._detector.detect(image)

                if detection.found:
                    print(f"[UseCase] L0 detected: {detection.label} (conf={detection.confidence:.2f})", flush=True)
                else:
                    print(f"[UseCase] L0: No object detected, but continuing to L1...", flush=True)

            except Exception as e:
                print(f"[UseCase] ERROR in L0 detection: {e}", flush=True)
                print(f"[UseCase] Continuing to L1 despite L0 error...", flush=True)
                traceback.print_exc()
                # ไม่ raise - ให้ L1 ทำงานต่อ
        else:
            print(f"[UseCase] L0 disabled, skipping detection", flush=True)

        # Step 3: Preprocess for L1
        try:
            print(f"[UseCase] Preprocessing for L1...", flush=True)
            processed = self._processor.preprocess(image)
        except Exception as e:
            print(f"[UseCase] ERROR preprocessing: {e}", flush=True)
            traceback.print_exc()
            raise

        # Step 4: L1 - Material Classification (Trash-Net)
        try:
            print(f"[UseCase] Running L1 (Trash-Net) classification...", flush=True)
            result = self._classifier.classify(processed)
            print(f"[UseCase] L1 result: {result.category} ({result.confidence:.2%})", flush=True)
        except Exception as e:
            print(f"[UseCase] ERROR in L1 classification: {e}", flush=True)
            traceback.print_exc()
            raise

        # Add L0 info to metadata if detector was used
        if self._detector:
            if detection and detection.found:
                result.metadata["l0_detected"] = True
                result.metadata["l0_label"] = detection.label
                result.metadata["l0_confidence"] = detection.confidence
            else:
                result.metadata["l0_detected"] = False
                result.metadata["l0_label"] = None
                result.metadata["l0_confidence"] = 0.0

        print(f"[UseCase] Classification complete!", flush=True)
        return result
