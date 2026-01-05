# AURA - Solar Panel Detection Pipeline

## Overview
AURA is an automated solar panel detection and classification system that uses:
- **YOLOv11-m** for oriented bounding box detection
- **DeepLabV3+** for pixel-level segmentation
- **Two-tier buffer zones** (1200/2400 sqft) for rooftop PV classification

## Quick Start

### Installation
```bash
pip install -r environment_details/requirements.txt
```

### Usage
```bash
python pipeline_code/solar_classify.py <image_path> --scale 0.1088
```

### Example
```bash
python pipeline_code/solar_classify.py satellite_image.png --scale 0.1088
```

## Project Structure
```
my-app/
├── pipeline_code/           # Python source code
│   ├── solar_classify.py    # Main classification pipeline
│   ├── solar_pipeline.py    # Base pipeline utilities
│   └── download_satellite.py # Google Maps satellite downloader
├── environment_details/     # Environment configuration
│   ├── requirements.txt     # Python dependencies
│   ├── python_version.txt   # Python version
│   └── environment.yml      # Conda environment
├── trained_model/           # Trained model weights
│   ├── model.pt             # YOLOv11-m weights
│   └── deeplabv3_final.pth  # DeepLabV3+ weights
├── model_card/              # Model documentation
│   └── model_card.md        # Model specifications
├── prediction_files/        # JSON prediction outputs
│   ├── train/               # Training set predictions
│   └── test/                # Test set predictions
├── artefacts/               # Visualization outputs
│   ├── train/               # Training visualizations
│   └── test/                # Test visualizations
├── training_logs/           # Training metrics
│   └── logs.csv             # Training logs
└── README.md                # This file
```

## Pipeline Workflow

### 1. CLASSIFY
- Detects solar panels using YOLOv11-m with TTA
- Checks for PV presence within 1200 sqft (primary) or 2400 sqft (secondary) buffer

### 2. QUANTIFY
- Estimates panel area using:
  - Pixel counting method
  - Contour area method (cv2.contourArea)
  - Returns average of both methods

### 3. VERIFY
- Calculates Euclidean distance from panel centroid to buffer center
- Validates panel location

## Key Features
- **Test-Time Augmentation (TTA)**: 3x augmentation for robust detection
- **Soft-NMS**: Improved handling of overlapping detections
- **Color Filtering**: Rejects false positives (zebra crossings, white surfaces)
- **Dual Buffer Segmentation**: Uses both 1200 and 2400 sqft contexts
- **OBB-Constrained Masking**: Segmentation limited to detection bounding box

## Output
- Annotated image with:
  - Gray overlay outside buffer zone
  - Buffer zone circles (green: 1200 sqft, yellow: 2400 sqft)
  - Detection boxes (green: selected, red: others)
  - Segmentation mask (magenta overlay)
- Console output with area calculations

## License
MIT License

## Authors
AURA Team
