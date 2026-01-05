"""
Solar Panel Classification Pipeline
=====================================
Workflow:
  1. CLASSIFY: Determine if rooftop PV is present within 1200 sq.ft buffer.
               If not, check within 2400 sq.ft buffer.
  2. QUANTIFY: Estimate total area (m²) of panel with largest buffer overlap.
  3. VERIFY:   Calculate Euclidean distance (m) of panel centroid from center.
"""

# Suppress warnings before importing torch
import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to prevent GPU errors
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import numpy as np
import math
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONSTANTS (from fi_1 reference)
# ============================================================
SQFT_TO_SQM_CONVERSION = 0.092903
BUFFER_ZONE_1_SQFT = 1200  # Primary buffer
BUFFER_ZONE_2_SQFT = 2400  # Secondary buffer

# GSD Configuration (Ground Sample Distance)
GSD_CM_PER_PIXEL = 10.88
GSD_M_PER_PIXEL = GSD_CM_PER_PIXEL / 100  # 0.1088 m/pixel


# ============================================================
# BUFFER ZONE CALCULATIONS
# ============================================================
def calculate_buffer_radii(scale_m_per_pixel=GSD_M_PER_PIXEL):
    """
    Calculate buffer zone radii in pixels for 1200 and 2400 sq.ft areas.
    Uses circular buffer: Area = pi * r^2 => r = sqrt(Area / pi)
    """
    buffer_1_sqm = BUFFER_ZONE_1_SQFT * SQFT_TO_SQM_CONVERSION
    buffer_2_sqm = BUFFER_ZONE_2_SQFT * SQFT_TO_SQM_CONVERSION
    
    radius_1_m = math.sqrt(buffer_1_sqm / math.pi)
    radius_2_m = math.sqrt(buffer_2_sqm / math.pi)
    
    radius_1_px = radius_1_m / scale_m_per_pixel
    radius_2_px = radius_2_m / scale_m_per_pixel
    
    return {
        'primary': {'sqft': 1200, 'sqm': buffer_1_sqm, 'radius_m': radius_1_m, 'radius_px': radius_1_px},
        'secondary': {'sqft': 2400, 'sqm': buffer_2_sqm, 'radius_m': radius_2_m, 'radius_px': radius_2_px}
    }


def calculate_overlap_area(bbox, buffer_center, buffer_radius, scale_m_per_pixel):
    """
    Calculate overlap area between panel bbox and circular buffer.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        buffer_center: (cx, cy) center of buffer zone
        buffer_radius: radius in pixels
        scale_m_per_pixel: scale for conversion
    
    Returns:
        overlap_area_sqm: Overlap area in square meters
    """
    x1, y1, x2, y2 = bbox
    panel_cx, panel_cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Distance from panel center to buffer center
    dist = math.sqrt((panel_cx - buffer_center[0])**2 + (panel_cy - buffer_center[1])**2)
    
    if dist <= buffer_radius:
        # Panel center is inside buffer - full overlap estimate
        bbox_area_px = (x2 - x1) * (y2 - y1)
        overlap_area_sqm = bbox_area_px * (scale_m_per_pixel ** 2)
    elif dist <= buffer_radius + max(x2-x1, y2-y1)/2:
        # Partial overlap
        bbox_area_px = (x2 - x1) * (y2 - y1)
        overlap_ratio = max(0, 1 - (dist - buffer_radius) / (max(x2-x1, y2-y1)/2))
        overlap_area_sqm = bbox_area_px * overlap_ratio * (scale_m_per_pixel ** 2)
    else:
        overlap_area_sqm = 0.0
    
    return overlap_area_sqm


def calculate_euclidean_distance(bbox, center, scale_m_per_pixel):
    """
    Calculate Euclidean distance from panel centroid to buffer center.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        center: (cx, cy) buffer center (image center)
        scale_m_per_pixel: scale for conversion
    
    Returns:
        distance_m: Distance in meters
    """
    x1, y1, x2, y2 = bbox
    panel_cx, panel_cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    dist_px = math.sqrt((panel_cx - center[0])**2 + (panel_cy - center[1])**2)
    dist_m = dist_px * scale_m_per_pixel
    
    return dist_m


# ============================================================
# MAIN CLASSIFICATION WORKFLOW
# ============================================================
class SolarPanelClassifier:
    def __init__(self, 
                 model_path='F:/AURA/final/models/best.pt',
                 conf_threshold=0.25,
                 scale_m_per_pixel=GSD_M_PER_PIXEL):
        """
        Initialize the solar panel classifier.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detection
            scale_m_per_pixel: Ground sample distance (m/pixel)
        """
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.scale = scale_m_per_pixel
        self.buffers = calculate_buffer_radii(scale_m_per_pixel)
        
        # Initialize DeepLabV3+ for segmentation
        self._init_deeplab()
    
    def classify(self, image_path):
        """
        Run the complete classification workflow.
        
        Steps:
            1. CLASSIFY: Check PV presence in 1200 sq.ft buffer, then 2400 sq.ft
            2. QUANTIFY: Estimate area of panel with largest overlap
            3. VERIFY: Calculate Euclidean distance from center
        
        Args:
            image_path: Path to satellite image
        
        Returns:
            result: Dictionary with classification results
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        buffer_center = (w / 2, h / 2)  # Image center
        
        print(f"Image: {image_path}")
        print(f"Size: {w}x{h} pixels")
        print(f"Scale: {self.scale} m/pixel")
        print(f"Buffer center: {buffer_center}")
        print(f"\nBuffer radii:")
        print(f"  Primary (1200 sqft): {self.buffers['primary']['radius_px']:.1f} px ({self.buffers['primary']['radius_m']:.2f} m)")
        print(f"  Secondary (2400 sqft): {self.buffers['secondary']['radius_px']:.1f} px ({self.buffers['secondary']['radius_m']:.2f} m)")
        
        # Run YOLO detection with TTA and multi-scale inference
        print(f"\n{'='*60}")
        print("STEP 1: CLASSIFY - Detecting solar panels (Enhanced)...")
        print(f"{'='*60}")
        
        detections = self._enhanced_yolo_detection(img, buffer_center)
        
        print(f"Found {len(detections)} panel detections")
        
        # --------------------------------------------------------
        # STEP 1: CLASSIFY - Two-tier buffer check
        # --------------------------------------------------------
        pv_present = False
        buffer_used = None
        relevant_buffer = None
        
        # Check primary buffer (1200 sqft) first
        panels_in_primary = [d for d in detections if d['overlap_primary_m2'] > 0]
        
        if panels_in_primary:
            pv_present = True
            buffer_used = 'primary'
            relevant_buffer = self.buffers['primary']
            print(f"\n✅ PV PRESENT in PRIMARY buffer (1200 sqft)")
            print(f"   {len(panels_in_primary)} panel(s) overlap with primary buffer")
        else:
            # Check secondary buffer (2400 sqft)
            panels_in_secondary = [d for d in detections if d['overlap_secondary_m2'] > 0]
            
            if panels_in_secondary:
                pv_present = True
                buffer_used = 'secondary'
                relevant_buffer = self.buffers['secondary']
                print(f"\n✅ PV PRESENT in SECONDARY buffer (2400 sqft)")
                print(f"   {len(panels_in_secondary)} panel(s) overlap with secondary buffer")
            else:
                print(f"\n❌ NO PV within buffer zones")
        
        # --------------------------------------------------------
        # STEP 2: QUANTIFY - Find panel with largest overlap
        # --------------------------------------------------------
        print(f"\n{'='*60}")
        print("STEP 2: QUANTIFY - Estimating panel area...")
        print(f"{'='*60}")
        
        best_panel = None
        if pv_present:
            if buffer_used == 'primary':
                # Find panel with largest primary overlap
                best_panel = max(panels_in_primary, key=lambda d: d['overlap_primary_m2'])
                overlap_key = 'overlap_primary_m2'
            else:
                # Find panel with largest secondary overlap
                best_panel = max(panels_in_secondary, key=lambda d: d['overlap_secondary_m2'])
                overlap_key = 'overlap_secondary_m2'
            
            print(f"Best panel (largest overlap):")
            print(f"  Panel ID: {best_panel['id']}")
            print(f"  Total area: {best_panel['area_m2']:.2f} m² ({best_panel['area_m2'] * 10.7639:.2f} sqft)")
            print(f"  Overlap area: {best_panel[overlap_key]:.2f} m²")
            print(f"  Confidence: {best_panel['confidence']:.3f}")
        else:
            print("No panels to quantify")
        
        # --------------------------------------------------------
        # STEP 3: VERIFY - Calculate Euclidean distance
        # --------------------------------------------------------
        print(f"\n{'='*60}")
        print("STEP 3: VERIFY - Calculating distance from center...")
        print(f"{'='*60}")
        
        if best_panel:
            print(f"Distance of selected panel from center:")
            print(f"  Euclidean distance: {best_panel['distance_from_center_m']:.2f} m")
            print(f"  Panel centroid: ({(best_panel['bbox'][0]+best_panel['bbox'][2])/2:.1f}, {(best_panel['bbox'][1]+best_panel['bbox'][3])/2:.1f}) px")
            print(f"  Buffer center: {buffer_center}")
        else:
            print("No panel to verify")
        
        # --------------------------------------------------------
        # FINAL RESULTS
        # --------------------------------------------------------
        print(f"\n{'='*60}")
        print("FINAL CLASSIFICATION RESULTS")
        print(f"{'='*60}")
        
        result = {
            'pv_present': pv_present,
            'buffer_used': buffer_used,
            'buffer_sqft': relevant_buffer['sqft'] if relevant_buffer else None,
            'confidence_score': best_panel['confidence'] if best_panel else 0.0,
            'panel_area_m2': best_panel['area_m2'] if best_panel else 0.0,
            'panel_area_sqft': best_panel['area_m2'] * 10.7639 if best_panel else 0.0,
            'overlap_area_m2': best_panel[overlap_key] if best_panel else 0.0,
            'distance_from_center_m': best_panel['distance_from_center_m'] if best_panel else 0.0,
            'total_detections': len(detections),
            'all_detections': detections
        }
        
        print(f"PV Present: {result['pv_present']}")
        print(f"Buffer Used: {result['buffer_used']} ({result['buffer_sqft']} sqft)" if result['buffer_used'] else "Buffer Used: None")
        print(f"Confidence Score: {result['confidence_score']:.3f}")
        print(f"Panel Area: {result['panel_area_m2']:.2f} m² ({result['panel_area_sqft']:.2f} sqft)")
        print(f"Overlap Area: {result['overlap_area_m2']:.2f} m²")
        print(f"Distance from Center: {result['distance_from_center_m']:.2f} m")
        
        # Run DeepLabV3+ segmentation ONLY on the best panel (largest overlap)
        segmentation_mask, mask_area_m2 = self._segment_best_panel(img, best_panel)
        result['segmentation_mask'] = segmentation_mask
        result['mask_area_m2'] = mask_area_m2
        result['mask_area_sqft'] = mask_area_m2 * 10.7639
        
        print(f"\n📊 DeepLabV3+ Mask Area: {mask_area_m2:.2f} m² ({mask_area_m2 * 10.7639:.2f} sqft)")
        
        # Create and save visualization with mask
        output_path = self._create_visualization(img, detections, buffer_center, best_panel, buffer_used, image_path, segmentation_mask)
        result['output_image'] = output_path
        print(f"\n📁 Annotated output saved: {output_path}")
        
        return result
    
    def _enhanced_yolo_detection(self, img, buffer_center):
        """
        Enhanced YOLO detection with:
        - Test-Time Augmentation (TTA): original + flipped versions
        - Multi-scale inference
        - Improved NMS for better accuracy
        """
        h, w = img.shape[:2]
        all_boxes = []
        all_confs = []
        
        # TTA: Run inference on multiple augmented versions
        augmentations = [
            ('original', img),
            ('hflip', cv2.flip(img, 1)),
            ('vflip', cv2.flip(img, 0)),
        ]
        
        print(f"  Running TTA with {len(augmentations)} augmentations...")
        
        for aug_name, aug_img in augmentations:
            results = self.model(aug_img, conf=self.conf * 0.8, verbose=False)[0]  # Lower conf for TTA
            
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy().copy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Transform bbox back to original coordinates
                if aug_name == 'hflip':
                    bbox[0], bbox[2] = w - bbox[2], w - bbox[0]
                elif aug_name == 'vflip':
                    bbox[1], bbox[3] = h - bbox[3], h - bbox[1]
                
                all_boxes.append(bbox)
                all_confs.append(conf)
        
        # Apply custom NMS to merge overlapping boxes
        if len(all_boxes) > 0:
            all_boxes = np.array(all_boxes)
            all_confs = np.array(all_confs)
            
            # Custom Soft-NMS
            final_boxes, final_confs = self._soft_nms(all_boxes, all_confs, iou_thresh=0.5)
            print(f"  After Soft-NMS: {len(final_boxes)} detections (from {len(all_boxes)} raw)")
        else:
            final_boxes, final_confs = [], []
        
        # Build detection list with overlap calculations
        detections = []
        for i, (bbox, conf) in enumerate(zip(final_boxes, final_confs)):
            # FALSE POSITIVE FILTER: Color + Aspect Ratio + Confidence
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip if invalid bbox
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract ROI for color analysis
            roi = img[y1:y2, x1:x2]
            
            # Filter 1: Color Analysis - Reject white/gray regions (zebra crossings)
            if not self._is_solar_panel_color(roi):
                continue
            
            # Filter 2: Aspect Ratio - Solar panels typically 1:1 to 3:1
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            if aspect_ratio > 4.0:  # Too elongated, likely not a panel
                continue
            
            # Filter 3: Minimum confidence (higher threshold after TTA)
            if conf < 0.15:  # Slightly higher threshold
                continue
            
            area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_m2 = area_px * (self.scale ** 2)
            
            overlap_1 = calculate_overlap_area(
                bbox, buffer_center, 
                self.buffers['primary']['radius_px'], 
                self.scale
            )
            overlap_2 = calculate_overlap_area(
                bbox, buffer_center,
                self.buffers['secondary']['radius_px'],
                self.scale
            )
            distance_m = calculate_euclidean_distance(bbox, buffer_center, self.scale)
            
            detections.append({
                'id': len(detections),  # Use sequential ID after filtering
                'bbox': bbox.tolist(),
                'confidence': conf,
                'area_m2': area_m2,
                'overlap_primary_m2': overlap_1,
                'overlap_secondary_m2': overlap_2,
                'distance_from_center_m': distance_m
            })
        
        print(f"  After color/aspect filtering: {len(detections)} valid panels")
        return detections
    
    def _is_solar_panel_color(self, roi):
        """
        Check if ROI has solar panel color characteristics.
        Solar panels are typically dark blue/black with low brightness.
        AGGRESSIVELY rejects white/gray regions like zebra crossings.
        """
        if roi.size == 0:
            return False
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Get mean values
        mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
        
        # Get color statistics
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        std_gray = np.std(gray)
        max_gray = np.max(gray)
        min_gray = np.min(gray)
        
        # Calculate percentage of white/bright pixels
        white_pixels = np.sum(gray > 180) / gray.size
        bright_pixels = np.sum(gray > 150) / gray.size
        
        # AGGRESSIVE REJECTION CRITERIA:
        
        # 1. Any significant white content (zebra crossings have white stripes)
        if white_pixels > 0.15:  # More than 15% white pixels
            return False
        
        # 2. Too much bright content overall
        if bright_pixels > 0.40:  # More than 40% bright pixels
            return False
        
        # 3. Too bright on average
        if mean_gray > 140 or mean_v > 160:
            return False
        
        # 4. High contrast with bright areas (zebra pattern)
        if std_gray > 50 and max_gray > 200:
            return False
        
        # 5. Low saturation with brightness (gray/white surfaces)
        if mean_s < 25 and mean_v > 120:
            return False
        
        # 6. Very wide dynamic range (white + dark = zebra)
        if (max_gray - min_gray) > 150 and max_gray > 200:
            return False
        
        # SOLAR PANEL ACCEPTANCE: Must be predominantly dark
        # Solar panels are dark blue/black (mean_gray typically < 100)
        if mean_gray > 130:
            return False
        
        return True
    
    def _soft_nms(self, boxes, scores, iou_thresh=0.5, sigma=0.5):
        """
        Soft-NMS implementation for better handling of overlapping detections.
        Instead of removing overlapping boxes, reduces their confidence.
        """
        if len(boxes) == 0:
            return [], []
        
        # Convert to list for modification
        boxes = boxes.tolist()
        scores = scores.tolist()
        
        picked_boxes = []
        picked_scores = []
        
        while len(boxes) > 0:
            # Find box with highest score
            max_idx = np.argmax(scores)
            max_box = boxes[max_idx]
            max_score = scores[max_idx]
            
            picked_boxes.append(max_box)
            picked_scores.append(max_score)
            
            # Remove from lists
            del boxes[max_idx]
            del scores[max_idx]
            
            # Apply soft suppression to remaining boxes
            i = 0
            while i < len(boxes):
                iou = self._calculate_iou(max_box, boxes[i])
                
                if iou > iou_thresh:
                    # Soft suppression: reduce score based on overlap
                    scores[i] *= np.exp(-(iou * iou) / sigma)
                    
                    if scores[i] < self.conf * 0.5:  # Remove if score too low
                        del boxes[i]
                        del scores[i]
                    else:
                        i += 1
                else:
                    i += 1
        
        return np.array(picked_boxes), np.array(picked_scores)
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _create_visualization(self, img, detections, buffer_center, best_panel, buffer_used, image_path, segmentation_mask=None):
        """Create annotated visualization with buffer zones, detections, and segmentation mask."""
        overlay = img.copy()
        h, w = img.shape[:2]
        
        # Create gray layer outside the selected buffer zone
        gray_layer = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_layer = cv2.cvtColor(gray_layer, cv2.COLOR_GRAY2BGR)
        
        # Determine which buffer is used and create mask for that zone
        if buffer_used == 'primary':
            buffer_radius = int(self.buffers['primary']['radius_px'])
        elif buffer_used == 'secondary':
            buffer_radius = int(self.buffers['secondary']['radius_px'])
        else:
            # No overlap - use secondary (larger) buffer for gray mask so both circles are visible
            buffer_radius = int(self.buffers['secondary']['radius_px'])
        
        # Create circular mask for the buffer zone
        buffer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(buffer_mask, (int(buffer_center[0]), int(buffer_center[1])), 
                   buffer_radius, 255, -1)
        
        # Apply gray effect outside buffer zone
        # Inside buffer = original color, Outside buffer = gray
        overlay = np.where(buffer_mask[:, :, np.newaxis] == 255, overlay, gray_layer)
        
        # Apply segmentation mask as semi-transparent magenta overlay
        if segmentation_mask is not None and np.any(segmentation_mask):
            mask_colored = np.zeros_like(img)
            mask_colored[:, :, 0] = segmentation_mask  # Blue
            mask_colored[:, :, 2] = segmentation_mask  # Red (Blue + Red = Magenta)
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # Draw contours around mask regions
            contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 255), 2)  # Magenta contours
        
        # Draw buffer zone circle(s)
        if buffer_used == 'primary':
            cv2.circle(overlay, (int(buffer_center[0]), int(buffer_center[1])), 
                       buffer_radius, (0, 255, 0), 2)  # Green for primary
        elif buffer_used == 'secondary':
            cv2.circle(overlay, (int(buffer_center[0]), int(buffer_center[1])), 
                       buffer_radius, (0, 255, 255), 2)  # Yellow for secondary
        else:
            # No panel overlaps - draw BOTH buffer circles
            radius_1 = int(self.buffers['primary']['radius_px'])
            radius_2 = int(self.buffers['secondary']['radius_px'])
            cv2.circle(overlay, (int(buffer_center[0]), int(buffer_center[1])), 
                       radius_1, (0, 255, 0), 2)  # Green for primary (1200 sqft)
            cv2.circle(overlay, (int(buffer_center[0]), int(buffer_center[1])), 
                       radius_2, (0, 255, 255), 2)  # Yellow for secondary (2400 sqft)
        
        # Draw center point
        cv2.circle(overlay, (int(buffer_center[0]), int(buffer_center[1])), 
                   5, (0, 0, 255), -1)  # Red center
        
        # Draw all detections
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            
            # Check if this is the best panel
            is_best = (best_panel and det['id'] == best_panel['id'])
            
            if is_best:
                # Best panel - thick green box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(overlay, "SELECTED", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Other panels - red box (no confidence text)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # Add legend (no confidence, add panel count)
        y_pos = 25
        cv2.putText(overlay, f"PV Present: {'YES' if best_panel else 'NO'}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if best_panel else (0, 0, 255), 2)
        y_pos += 25
        cv2.putText(overlay, f"Total Panels: {len(detections)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos += 25
        # Show buffer as sqft value
        if buffer_used == 'primary':
            buffer_text = "1200 sqft"
        elif buffer_used == 'secondary':
            buffer_text = "2400 sqft"
        else:
            buffer_text = "None"
        cv2.putText(overlay, f"Buffer Used: {buffer_text}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save output
        output_dir = Path("F:/AURA/final/classify_output")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{Path(image_path).stem}_classified.jpg"
        cv2.imwrite(str(output_path), overlay)
        
        return str(output_path)
    
    def _init_deeplab(self):
        """Initialize DeepLabV3+ model for segmentation with trained weights."""
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision import transforms
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Path to trained DeepLabV3+ model
        model_path = 'F:/AURA/final/models/deeplabv3_final.pth'
        
        # Load architecture with aux_classifier (matching training config)
        self.deeplab = deeplabv3_resnet101(pretrained=False, aux_loss=True)
        self.deeplab.classifier[-1] = torch.nn.Conv2d(256, 2, 1)  # 2 classes
        self.deeplab.aux_classifier[-1] = torch.nn.Conv2d(256, 2, 1)  # aux head also 2 classes
        
        # Load trained weights
        if Path(model_path).exists():
            print(f"Loading trained DeepLabV3+ from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.deeplab.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.deeplab.load_state_dict(checkpoint)
            print("✅ Trained DeepLabV3+ weights loaded successfully!")
        else:
            print(f"⚠️ Model not found: {model_path}, using random weights")
        
        self.deeplab.to(self.device)
        self.deeplab.eval()
        
        self.deeplab_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print(f"DeepLabV3+ initialized on {self.device}")
    
    def _segment_roi(self, roi):
        """
        Run DeepLabV3+ segmentation on a region of interest.
        Uses Test-Time Augmentation (TTA) for improved accuracy.
        """
        import torch
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Test-Time Augmentation: run on original + flipped versions
        predictions = []
        
        # Original
        input_tensor = self.deeplab_transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.deeplab(input_tensor)['out']
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()  # Get solar panel probability
        predictions.append(pred)
        
        # Horizontal flip
        rgb_hflip = cv2.flip(rgb, 1)
        input_tensor = self.deeplab_transform(rgb_hflip).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.deeplab(input_tensor)['out']
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        predictions.append(cv2.flip(pred, 1))  # Flip back
        
        # Vertical flip
        rgb_vflip = cv2.flip(rgb, 0)
        input_tensor = self.deeplab_transform(rgb_vflip).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.deeplab(input_tensor)['out']
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        predictions.append(cv2.flip(pred, 0))  # Flip back
        
        # Average TTA predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Apply threshold
        mask = (avg_pred > 0.3).astype(np.uint8) * 255  # Lower threshold for better recall
        
        return mask
    
    def _postprocess_mask(self, mask):
        """Apply improved post-processing to refine segmentation mask."""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Smaller kernel for fine details
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Morphological opening (remove small noise)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Morphological closing (fill small holes)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Binary thresholding
        _, binary = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
        
        # Connected components - filter very small regions only
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        MIN_AREA = 50  # Lower threshold to keep smaller panels
        refined_mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_AREA:
                refined_mask[labels == i] = 255
        
        return refined_mask
    
    def _segment_best_panel(self, img, best_panel):
        """
        Run DeepLabV3+ segmentation on the panel with largest overlap.
        Uses BOTH 1200 sqft and 2400 sqft buffer zones for better context.
        Combines masks from both buffer sizes using TTA-style averaging.
        
        Args:
            img: Input image
            best_panel: Detection dict with bbox of the best panel
        
        Returns:
            full_mask: Binary mask for the best panel
            mask_area_m2: Calculated area in square meters
        """
        h, w = img.shape[:2]
        
        print(f"\n{'='*60}")
        print(f"SAHI SLICING - Dual Buffer Segmentation")
        print(f"{'='*60}")
        
        if best_panel is None:
            print("No panel to segment!")
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        # Get panel bbox
        x1, y1, x2, y2 = [int(v) for v in best_panel['bbox']]
        panel_cx = (x1 + x2) // 2
        panel_cy = (y1 + y2) // 2
        
        print(f"Best Panel ID: {best_panel['id']}")
        print(f"  Bbox: ({x1}, {y1}) to ({x2}, {y2})")
        
        # SAHI-style slicing parameters
        SLICE_SIZE = 128
        STRIDE = 64
        
        masks_combined = []
        
        # Process with BOTH buffer sizes for better context
        buffer_sizes = [
            ('1200 sqft', int(self.buffers['primary']['radius_px'])),
            ('2400 sqft', int(self.buffers['secondary']['radius_px']))
        ]
        
        for buffer_name, buffer_radius in buffer_sizes:
            print(f"\n  Processing with {buffer_name} buffer (radius={buffer_radius}px)...")
            
            # Calculate ROI based on buffer
            roi_half = buffer_radius
            px1 = max(0, panel_cx - roi_half)
            py1 = max(0, panel_cy - roi_half)
            px2 = min(w, panel_cx + roi_half)
            py2 = min(h, panel_cy + roi_half)
            
            # Extract panel ROI
            panel_roi = img[py1:py2, px1:px2].copy()
            roi_h, roi_w = panel_roi.shape[:2]
            
            print(f"    ROI: ({px1}, {py1}) to ({px2}, {py2}), size: {roi_w}x{roi_h}")
            
            # Segment the ROI
            if roi_h < SLICE_SIZE or roi_w < SLICE_SIZE:
                panel_mask = self._segment_roi(panel_roi)
            else:
                # SAHI slicing for larger ROIs
                roi_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
                count_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
                
                for y in range(0, max(1, roi_h - SLICE_SIZE + 1), STRIDE):
                    for x in range(0, max(1, roi_w - SLICE_SIZE + 1), STRIDE):
                        slice_img = panel_roi[y:y+SLICE_SIZE, x:x+SLICE_SIZE]
                        slice_mask = self._segment_roi(slice_img)
                        
                        sh, sw = slice_mask.shape[:2]
                        roi_mask[y:y+sh, x:x+sw] += slice_mask.astype(np.float32) / 255.0
                        count_mask[y:y+sh, x:x+sw] += 1
                
                count_mask[count_mask == 0] = 1
                roi_mask = roi_mask / count_mask
                panel_mask = ((roi_mask > 0.5) * 255).astype(np.uint8)
            
            # Create full-size mask and place ROI mask
            buffer_full_mask = np.zeros((h, w), dtype=np.uint8)
            buffer_full_mask[py1:py2, px1:px2] = panel_mask[:py2-py1, :px2-px1]
            
            masks_combined.append(buffer_full_mask.astype(np.float32) / 255.0)
        
        # Combine masks from both buffers (average)
        combined_mask = np.mean(masks_combined, axis=0)
        final_mask = ((combined_mask > 0.3) * 255).astype(np.uint8)  # Lower threshold
        
        # Post-process combined mask
        final_mask = self._postprocess_mask(final_mask)
        
        # CONSTRAIN MASK TO OBB BOUNDING BOX ONLY
        # Create a mask that is only valid inside the OBB
        obb_mask = np.zeros((h, w), dtype=np.uint8)
        obb_mask[y1:y2, x1:x2] = 255  # Only the OBB area
        
        # Apply OBB constraint - mask only where OBB overlaps
        final_mask = cv2.bitwise_and(final_mask, obb_mask)
        
        # ============================================================
        # DUAL AREA CALCULATION: Pixel Count + Polygon Contour
        # ============================================================
        
        # Method 1: Pixel Counting
        mask_pixels = np.count_nonzero(final_mask)
        pixel_area_m2 = mask_pixels * (self.scale ** 2)
        
        # Method 2: Polygon Contour Area (more accurate for irregular shapes)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area_px = 0
        for contour in contours:
            contour_area_px += cv2.contourArea(contour)
        contour_area_m2 = contour_area_px * (self.scale ** 2)
        
        # Average of both methods for best estimate
        mask_area_m2 = (pixel_area_m2 + contour_area_m2) / 2
        
        print(f"\n✅ Segmentation complete (constrained to OBB)!")
        print(f"   OBB: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"   Pixel Count Area: {pixel_area_m2:.2f} m² ({pixel_area_m2 * 10.7639:.2f} sqft)")
        print(f"   Contour Area:     {contour_area_m2:.2f} m² ({contour_area_m2 * 10.7639:.2f} sqft)")
        print(f"   Average Area:     {mask_area_m2:.2f} m² ({mask_area_m2 * 10.7639:.2f} sqft)")
        
        return final_mask, mask_area_m2


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solar Panel Classification")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--model", default="F:/AURA/final/models/best.pt", help="YOLO model path")
    parser.add_argument("--scale", type=float, default=GSD_M_PER_PIXEL, help="Meters per pixel")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    classifier = SolarPanelClassifier(
        model_path=args.model,
        conf_threshold=args.conf,
        scale_m_per_pixel=args.scale
    )
    
    result = classifier.classify(args.image)
