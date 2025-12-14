from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..enums.trash_category import TrashCategory, get_bin_info


@dataclass
class ClassificationResult:
    """Classification result entity"""
    category: Optional[TrashCategory]  # Can be None if L0 rejects
    confidence: float
    model_used: str
    sub_result: Optional['ClassificationResult'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_rejected(self) -> bool:
        """Check if this result is a rejection (L0 didn't find object)"""
        return self.category is None or self.metadata.get("rejected", False)

    def is_confident(self, threshold: float) -> bool:
        """Check if confidence meets threshold"""
        return self.confidence >= threshold

    def get_bin_info(self) -> dict:
        """Get bin information for this category"""
        if self.category is None:
            return {"number": 0, "label_th": "ไม่ทราบ", "label_en": "Unknown"}
        return get_bin_info(self.category)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        # Handle L0 rejection case - check category is None directly
        if self.category is None or self.metadata.get("rejected", False):
            return {
                "category": "rejected",
                "confidence": round(self.confidence, 4),
                "model_used": self.model_used,
                "bin_number": 0,
                "bin_label": "ไม่ทราบ",
                "message": self.metadata.get("message", "ไม่สามารถจำแนกได้"),
                "rejected": True,
                "reject_reason": self.metadata.get("reason", "unknown"),
            }

        bin_info = self.get_bin_info()

        # Safe access to category.value since we know it's not None here
        result = {
            "category": self.category.value,
            "confidence": round(self.confidence, 4),
            "model_used": self.model_used,
            "bin_number": bin_info["number"],
            "bin_label": bin_info["label_th"],
            "message": f"ทิ้งช่อง {bin_info['number']} - {bin_info['label_th']}",
        }

        if self.sub_result and self.sub_result.category is not None:
            result["sub_category"] = self.sub_result.category.value
            result["sub_confidence"] = round(self.sub_result.confidence, 4)

        return result
