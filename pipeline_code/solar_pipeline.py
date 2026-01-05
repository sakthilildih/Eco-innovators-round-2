"""
Solar Panel Detection & Segmentation Pipeline
==============================================
6-Step Pipeline:
  Step 1: Preprocessing (Grayscale + CLAHE)
  Step 2: YOLO Detection (Grayscale image)
  Step 3: Buffer Zone (4000 sq.ft + Gaussian)
  Step 4: DeepLabV3+ Segmentation (Color image)
  Step 5: Post-Processing (Canny, Morphology, Threshold)
  Step 6: Area Calculation (Pixel to Metre)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# STEP 1: PREPROCESSING
# ============================================================
def step1_preprocess(image):
    """
    Apply grayscale conversion and CLAHE contrast enhancement.
    
    Args:
        image: BGR input image
    
    Returns:
        enhanced: Grayscale enhanced image
        original: Original color image (for DeepLabV3+)
    """
    # Keep original color for DeepLabV3+
    original_color = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to 3-channel for YOLO (grayscale in 3 channels)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_3ch, original_color


# ============================================================
# STEP 2: YOLO DETECTION (GRAYSCALE)
# ============================================================
class YOLODetector:
    def __init__(self, model_path='F:/AURA/final/models/best.pt', conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
    
    def detect(self, grayscale_image):
        """
        Run YOLO detection on grayscale enhanced image.
        
        Args:
            grayscale_image: 3-channel grayscale image
        
        Returns:
            detections: List of (bbox, confidence, mask)
        """
        results = self.model(grayscale_image, conf=self.conf)[0]
        
        detections = []
        for i, box in enumerate(results.boxes):
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            
            # Get mask if available (segmentation model)
            mask = None
            if results.masks is not None and len(results.masks) > i:
                mask = results.masks[i].data.cpu().numpy().squeeze()
            
            detections.append({
                'bbox': bbox,  # (x1, y1, x2, y2)
                'confidence': conf,
                'yolo_mask': mask
            })
        
        return detections


# ============================================================
# STEP 3: TWO-TIER BUFFER ZONE (1200/2400 sq.ft + GAUSSIAN)
# ============================================================

# Constants from reference code (fi_1/step3_buffer_filter.py)
SQFT_TO_SQM_CONVERSION = 0.092903
BUFFER_ZONE_1_SQFT = 1200  # Primary buffer
BUFFER_ZONE_2_SQFT = 2400  # Secondary buffer

# GSD Configuration (Ground Sample Distance)
# At zoom 20, GSD is approximately 10.88 cm/pixel
GSD_CM_PER_PIXEL = 10.88
GSD_M_PER_PIXEL = GSD_CM_PER_PIXEL / 100  # 0.1088 m/pixel


def calculate_buffer_radii(gsd: float = None, scale_m_per_pixel: float = None):
    """
    Calculate buffer zone radii in pixels for 1200 and 2400 sq.ft areas.
    Uses circular buffer zones: Area = pi * r^2 => r = sqrt(Area / pi)
    
    Args:
        gsd: Ground Sample Distance in m/pixel (default: 0.1088)
        scale_m_per_pixel: Alternative scale parameter
    
    Returns:
        (radius_1_pixels, radius_2_pixels, info_dict)
    """
    import math
    
    # Use provided scale or default GSD
    if scale_m_per_pixel is not None:
        gsd = scale_m_per_pixel
    elif gsd is None:
        gsd = GSD_M_PER_PIXEL
    
    # Convert sq.ft to sq.m
    buffer_zone_1_sqm = BUFFER_ZONE_1_SQFT * SQFT_TO_SQM_CONVERSION  # ~111.5 m²
    buffer_zone_2_sqm = BUFFER_ZONE_2_SQFT * SQFT_TO_SQM_CONVERSION  # ~223 m²
    
    # Calculate radius: Area = pi * r^2 => r = sqrt(Area / pi)
    radius_1_m = math.sqrt(buffer_zone_1_sqm / math.pi)  # ~5.96 m
    radius_2_m = math.sqrt(buffer_zone_2_sqm / math.pi)  # ~8.43 m
    
    # Convert from meters to pixels
    radius_1_pixels = radius_1_m / gsd
    radius_2_pixels = radius_2_m / gsd
    
    info = {
        'primary': {
            'sqft': BUFFER_ZONE_1_SQFT,
            'sqm': buffer_zone_1_sqm,
            'radius_m': radius_1_m,
            'radius_pixels': radius_1_pixels
        },
        'secondary': {
            'sqft': BUFFER_ZONE_2_SQFT,
            'sqm': buffer_zone_2_sqm,
            'radius_m': radius_2_m,
            'radius_pixels': radius_2_pixels
        },
        'gsd_m_per_pixel': gsd
    }
    
    return radius_1_pixels, radius_2_pixels, info


def step3_buffer_zone(color_image, bbox, scale_m_per_pixel=0.3, use_secondary=False):
    """
    Create circular buffer zone around detection and apply Gaussian filter.
    
    Two-tier buffer system (from fi_1 reference code):
        Primary:   1200 sq.ft = 111.5 m² (radius ~5.96m)
        Secondary: 2400 sq.ft = 223 m²   (radius ~8.43m)
    
    Args:
        color_image: Original COLOR image
        bbox: (x1, y1, x2, y2) from YOLO
        scale_m_per_pixel: Meters per pixel for scale conversion
        use_secondary: If True, use 2400 sq.ft buffer; else 1200 sq.ft
    
    Returns:
        roi: Denoised color ROI
        coords: (bx1, by1, bx2, by2) buffer coordinates
        buffer_info: Dictionary with buffer size calculations
    """
    # Calculate both buffer radii
    radius_1, radius_2, buffer_calc_info = calculate_buffer_radii(
        scale_m_per_pixel=scale_m_per_pixel
    )
    
    # Select which buffer to use
    if use_secondary:
        buffer_radius = radius_2
        buffer_sqft = BUFFER_ZONE_2_SQFT
        buffer_type = 'secondary'
    else:
        buffer_radius = radius_1
        buffer_sqft = BUFFER_ZONE_1_SQFT
        buffer_type = 'primary'
    
    # Get detection center
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Create square bounding box around circular buffer
    buffer_size_pixels = int(buffer_radius * 2)  # Diameter
    half = buffer_size_pixels // 2
    h, w = color_image.shape[:2]
    
    bx1 = max(0, cx - half)
    by1 = max(0, cy - half)
    bx2 = min(w, cx + half)
    by2 = min(h, cy + half)
    
    # Extract COLOR ROI (for DeepLabV3+)
    roi = color_image[by1:by2, bx1:bx2].copy()
    
    # Apply Gaussian filter for denoising
    denoised = cv2.GaussianBlur(roi, (5, 5), 0)
    
    # Buffer info for debugging/logging
    buffer_info = {
        'buffer_type': buffer_type,
        'buffer_sqft': buffer_sqft,
        'buffer_sqm': buffer_sqft * SQFT_TO_SQM_CONVERSION,
        'buffer_radius_m': buffer_calc_info[buffer_type]['radius_m'],
        'buffer_radius_pixels': buffer_radius,
        'buffer_size_pixels': buffer_size_pixels,
        'scale_m_per_pixel': scale_m_per_pixel,
        'primary_radius': radius_1,
        'secondary_radius': radius_2
    }
    
    return denoised, (bx1, by1, bx2, by2), buffer_info


# ============================================================
# STEP 4: DeepLabV3+ SEGMENTATION (COLOR)
# ============================================================
class DeepLabSegmenter:
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize DeepLabV3+ for segmentation.
        Uses pretrained model or loads custom weights.
        """
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision import transforms
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.classifier[-1] = torch.nn.Conv2d(256, 2, 1)  # 2 classes
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def segment(self, color_roi, slice_size=256, stride=128):
        """
        Dynamic slicing segmentation on COLOR image.
        
        Args:
            color_roi: Color ROI from buffer zone
            slice_size: Window size (default: 256)
            stride: Overlap stride (default: 128)
        
        Returns:
            mask: Binary segmentation mask
        """
        h, w = color_roi.shape[:2]
        
        # If ROI is smaller than slice size, process directly
        if h < slice_size or w < slice_size:
            return self._segment_single(color_roi)
        
        # Full mask with count for averaging
        full_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)
        
        # Slide window across ROI
        for y in range(0, h - slice_size + 1, stride):
            for x in range(0, w - slice_size + 1, stride):
                slice_img = color_roi[y:y+slice_size, x:x+slice_size]
                pred = self._segment_single(slice_img)
                
                full_mask[y:y+slice_size, x:x+slice_size] += pred
                count_mask[y:y+slice_size, x:x+slice_size] += 1
        
        # Average overlapping regions
        count_mask[count_mask == 0] = 1
        full_mask = full_mask / count_mask
        
        # Threshold to binary
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def _segment_single(self, img):
        """Segment a single image patch."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return pred.astype(np.float32)


# ============================================================
# STEP 5: POST-PROCESSING
# ============================================================
def step5_postprocess(mask):
    """
    Apply post-processing to refine segmentation mask.
    
    Operations:
        1. Canny edge detection
        2. Morphological opening (remove noise)
        3. Morphological closing (fill holes)
        4. Binary thresholding
        5. Connected component filtering
    
    Args:
        mask: Input binary mask
    
    Returns:
        refined_mask: Cleaned binary mask
    """
    # Ensure uint8
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # 1. Morphological opening (remove small noise)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Morphological closing (fill small holes)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Binary thresholding
    _, binary = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Connected components - filter by area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    MIN_AREA = 100  # pixels
    refined_mask = np.zeros_like(binary)
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_AREA:
            refined_mask[labels == i] = 255
    
    # 5. Optional: Canny edge enhancement
    edges = cv2.Canny(refined_mask, 50, 150)
    
    return refined_mask, edges


# ============================================================
# STEP 6: AREA CALCULATION
# ============================================================
def step6_calculate_area(mask, scale_m_per_pixel=0.3):
    """
    Calculate real-world area from mask.
    
    Args:
        mask: Binary mask
        scale_m_per_pixel: Meters per pixel (default: 0.3 for Google Maps Z18)
    
    Returns:
        area_m2: Area in square meters
        area_sqft: Area in square feet
        panel_count: Number of panels detected
    """
    # Method 1: Pixel counting
    pixel_count = np.count_nonzero(mask)
    area_m2 = pixel_count * (scale_m_per_pixel ** 2)
    
    # Convert to sq.ft
    area_sqft = area_m2 * 10.7639
    
    # Method 2: Contour-based for individual panels
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    panel_count = len(contours)
    panel_areas = []
    
    for contour in contours:
        pixel_area = cv2.contourArea(contour)
        real_area = pixel_area * (scale_m_per_pixel ** 2)
        panel_areas.append(real_area)
    
    return {
        'total_area_m2': area_m2,
        'total_area_sqft': area_sqft,
        'panel_count': panel_count,
        'panel_areas_m2': panel_areas
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
class SolarPanelPipeline:
    def __init__(self, 
                 yolo_model_path='F:/AURA/final/models/best.pt',
                 deeplab_model_path=None,
                 conf_threshold=0.25,
                 scale_m_per_pixel=0.3):
        """
        Initialize the complete pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model
            deeplab_model_path: Path to DeepLabV3+ weights (optional)
            conf_threshold: YOLO confidence threshold
            scale_m_per_pixel: Meters per pixel (used for buffer zone & area calc)
        """
        self.yolo = YOLODetector(yolo_model_path, conf_threshold)
        self.deeplab = DeepLabSegmenter(deeplab_model_path)
        self.scale = scale_m_per_pixel
    
    def process(self, image_path, output_dir=None):
        """
        Run complete pipeline on an image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)
        
        Returns:
            results: Dictionary with masks, areas, and visualizations
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing: {image_path}")
        print(f"Image size: {img.shape}")
        
        # Step 1: Preprocessing
        print("Step 1: Preprocessing...")
        grayscale_enhanced, color_original = step1_preprocess(img)
        
        # Step 2: YOLO Detection
        print("Step 2: YOLO Detection...")
        detections = self.yolo.detect(grayscale_enhanced)
        print(f"  Found {len(detections)} detections")
        
        # Process each detection
        results = {
            'detections': [],
            'total_area_m2': 0,
            'total_area_sqft': 0,
            'panel_count': 0
        }
        
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        for i, det in enumerate(detections):
            print(f"\nProcessing detection {i+1}/{len(detections)}...")
            
            # Step 3: Buffer Zone
            print("  Step 3: Buffer Zone + Gaussian...")
            roi, coords, buffer_info = step3_buffer_zone(
                color_original, det['bbox'], self.scale
            )
            print(f"    Buffer: {buffer_info['buffer_type']} ({buffer_info['buffer_sqft']} sqft, r={buffer_info['buffer_radius_pixels']:.0f}px)")
            
            # Step 4: DeepLabV3+ Segmentation
            print("  Step 4: DeepLabV3+ Segmentation...")
            local_mask = self.deeplab.segment(roi)
            
            # Step 5: Post-Processing
            print("  Step 5: Post-Processing...")
            refined_mask, edges = step5_postprocess(local_mask)
            
            # Step 6: Area Calculation
            print("  Step 6: Area Calculation...")
            area_info = step6_calculate_area(refined_mask, self.scale)
            
            # Place mask in full image
            bx1, by1, bx2, by2 = coords
            full_mask[by1:by2, bx1:bx2] = np.maximum(
                full_mask[by1:by2, bx1:bx2],
                refined_mask[:by2-by1, :bx2-bx1]
            )
            
            # Accumulate results
            results['detections'].append({
                'bbox': det['bbox'].tolist(),
                'confidence': det['confidence'],
                'area_m2': area_info['total_area_m2'],
                'area_sqft': area_info['total_area_sqft'],
                'panel_count': area_info['panel_count']
            })
            
            results['total_area_m2'] += area_info['total_area_m2']
            results['total_area_sqft'] += area_info['total_area_sqft']
            results['panel_count'] += area_info['panel_count']
        
        results['full_mask'] = full_mask
        
        # Create visualization
        overlay = self._create_overlay(color_original, full_mask, detections)
        results['overlay'] = overlay
        
        # Save if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            name = Path(image_path).stem
            cv2.imwrite(str(output_dir / f"{name}_mask.png"), full_mask)
            cv2.imwrite(str(output_dir / f"{name}_overlay.jpg"), overlay)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"PIPELINE RESULTS")
        print(f"{'='*50}")
        print(f"Detections: {len(detections)}")
        print(f"Panel Count: {results['panel_count']}")
        print(f"Total Area: {results['total_area_m2']:.2f} m²")
        print(f"Total Area: {results['total_area_sqft']:.2f} sq.ft")
        
        return results
    
    def _create_overlay(self, image, mask, detections):
        """Create visualization overlay."""
        overlay = image.copy()
        
        # Apply mask as semi-transparent green
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 1] = mask  # Green channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"{det['confidence']:.2f}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return overlay


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solar Panel Detection Pipeline")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--yolo", default="F:/AURA/final/models/best.pt", 
                       help="YOLO model path")
    parser.add_argument("--scale", type=float, default=0.3, 
                       help="Meters per pixel")
    parser.add_argument("--conf", type=float, default=0.25, 
                       help="YOLO confidence threshold")
    
    args = parser.parse_args()
    
    pipeline = SolarPanelPipeline(
        yolo_model_path=args.yolo,
        conf_threshold=args.conf,
        scale_m_per_pixel=args.scale
    )
    
    results = pipeline.process(args.image, args.output)
