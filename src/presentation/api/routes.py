from fastapi import APIRouter, HTTPException
import httpx
import traceback
import sys

from .schemas import ClassifyRequest, ClassifyResponse, ErrorResponse

router = APIRouter()

# Global reference to use case (set by main.py)
classify_use_case = None


def set_use_case(use_case):
    """Set the classification use case (called by main.py)"""
    global classify_use_case
    classify_use_case = use_case


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Server Error"},
    }
)
async def classify_trash(request: ClassifyRequest):
    """
    Classify trash image from URL

    - **image_url**: URL of the image to classify (from R2 storage)
    - **trash_id**: Optional ID for tracking

    Returns classification result with:
    - category: cardboard, glass, metal, paper, plastic, trash
    - confidence: 0.0 - 1.0
    - bin_number: 1-6
    - bin_label: Thai label
    - message: Full message for display
    """
    if classify_use_case is None:
        raise HTTPException(
            status_code=500,
            detail="Classification service not initialized"
        )

    try:
        # Execute classification
        print(f"\n[API] Classifying image: {str(request.image_url)[:80]}...")

        result = classify_use_case.execute(str(request.image_url))
        result_dict = result.to_dict()

        # Check if L0 rejected
        if result_dict.get("rejected"):
            print(f"[API] L0 Rejected: {result_dict.get('reject_reason', 'unknown')}")
        else:
            print(f"[API] Result: {result_dict['category']} ({result_dict['confidence']:.2%})")

        return ClassifyResponse(
            success=True,
            category=result_dict.get("category", "unknown"),
            confidence=result_dict.get("confidence", 0.0),
            bin_number=result_dict.get("bin_number", 0),
            bin_label=result_dict.get("bin_label", "ไม่ทราบ"),
            message=result_dict.get("message", "ไม่สามารถจำแนกได้"),
            model_used=result_dict.get("model_used", "unknown"),
            sub_category=result_dict.get("sub_category"),
            sub_confidence=result_dict.get("sub_confidence"),
            trash_id=request.trash_id
        )

    except httpx.HTTPError as e:
        print(f"[API] HTTP Error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch image: {str(e)}"
        )

    except Exception as e:
        print(f"[API] Error: {e}", flush=True)
        print(f"[API] Full traceback:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )
