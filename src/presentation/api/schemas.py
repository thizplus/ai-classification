from pydantic import BaseModel, HttpUrl
from typing import Optional


class ClassifyRequest(BaseModel):
    """Request schema for classification"""
    image_url: HttpUrl
    trash_id: Optional[str] = None  # For tracking


class ClassifyResponse(BaseModel):
    """Response schema for classification"""
    success: bool
    category: str
    confidence: float
    bin_number: int
    bin_label: str
    message: str
    model_used: str
    sub_category: Optional[str] = None
    sub_confidence: Optional[float] = None
    trash_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    l0_enabled: bool = False
    l0_model: Optional[str] = None
