import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, set_use_case
from .schemas import HealthResponse
from src.infrastructure.models.trash_net import TrashNetClassifier
from src.infrastructure.models.yolo_detector import YOLODetector
from src.infrastructure.image.processor import ImageProcessor
from src.application.classify_trash import ClassifyTrashUseCase

# Global instances
classifier = None
detector = None
use_case = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global classifier, detector, use_case

    print("\n" + "=" * 50)
    print("Smart Trash AI Classification Service")
    print("=" * 50 + "\n")

    # Initialize L0 - YOLO Object Detector
    enable_l0 = os.getenv("ENABLE_L0_DETECTION", "true").lower() == "true"
    if enable_l0:
        print("[Startup] Loading L0 - YOLO Detector...")
        l0_threshold = float(os.getenv("L0_CONFIDENCE_THRESHOLD", "0.3"))
        detector = YOLODetector(threshold=l0_threshold)
    else:
        print("[Startup] L0 Detection disabled")
        detector = None

    # Initialize L1 - Trash-Net Classifier
    print("[Startup] Loading L1 - Trash-Net Classifier...")
    l1_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
    classifier = TrashNetClassifier(threshold=l1_threshold)

    # Initialize image processor
    image_processor = ImageProcessor()

    # Initialize use case with L0 + L1
    use_case = ClassifyTrashUseCase(
        classifier=classifier,
        image_processor=image_processor,
        detector=detector
    )

    # Set use case for routes
    set_use_case(use_case)

    print("\n[Startup] Service ready!")
    print(f"[Startup] L0 (YOLO) enabled: {enable_l0}")
    if enable_l0:
        print(f"[Startup] L0 threshold: {l0_threshold}")
    print(f"[Startup] L1 threshold: {l1_threshold}")
    print("=" * 50 + "\n")

    yield

    # Shutdown
    print("\n[Shutdown] Cleaning up...")


def create_app() -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="Smart Trash AI Classification",
        description="AI service for trash classification using Trash-Net model",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api", tags=["Classification"])

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint"""
        return {
            "message": "Smart Trash AI Classification Service",
            "version": "1.0.0",
            "docs": "/docs"
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Health check endpoint"""
        return HealthResponse(
            status="ok" if classifier is not None else "not_ready",
            model_loaded=classifier is not None,
            device=classifier._device if classifier else "unknown",
            l0_enabled=detector is not None,
            l0_model=detector.get_model_name() if detector else None
        )

    return app


# Create app instance
app = create_app()
