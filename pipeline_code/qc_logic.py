"""
Quality Control (QC) logic for determining if detections are verifiable.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def determine_qc_status(
    image_fetch_success: bool,
    detections: List[Dict],
    image_metadata: Dict = None,
    notes: str = ""
) -> tuple:
    """
    Determine the QC status for a detection result.
    Returns tuple of (status, reason).
    """
    if not image_fetch_success:
        return ("NOT_VERIFIABLE", "Image fetch failed")
    
    if image_metadata:
        if image_metadata.get("quality_issue"):
            return ("NOT_VERIFIABLE", image_metadata.get("quality_issue"))
        
        if image_metadata.get("resolution_warning"):
            return ("NOT_VERIFIABLE", "Low resolution")
    
    not_verifiable_keywords = [
        "cloud", "shadow", "occluded", "tree", "tank",
        "missing", "corrupted", "poor quality", "low resolution"
    ]
    
    if notes:
        notes_lower = notes.lower()
        for keyword in not_verifiable_keywords:
            if keyword in notes_lower:
                return ("NOT_VERIFIABLE", f"Quality issue: {keyword}")
    
    if detections and len(detections) > 0:
        return ("VERIFIABLE", None)
    
    return ("VERIFIABLE", None)


def check_image_quality(image_path: str) -> Dict:
    """
    Perform basic image quality checks.
    """
    import cv2
    
    result = {
        "quality_ok": True,
        "quality_issue": None,
        "resolution_warning": False
    }
    
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            result["quality_ok"] = False
            result["quality_issue"] = "Failed to read image"
            return result
        
        height, width = img.shape[:2]
        if height < 200 or width < 200:
            result["quality_ok"] = False
            result["quality_issue"] = "Image resolution too low"
            result["resolution_warning"] = True
            return result
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        
        if mean_brightness < 20:
            result["quality_ok"] = False
            result["quality_issue"] = "Image too dark"
        elif mean_brightness > 235:
            result["quality_ok"] = False
            result["quality_issue"] = "Image too bright (possible cloud cover)"
        
        variance = gray.var()
        if variance < 100:
            logger.warning(f"Low image variance ({variance:.1f}) - possible cloud cover or blur")
        
    except Exception as e:
        logger.warning(f"Error in quality check: {e}")
        result["quality_ok"] = True
    
    return result
