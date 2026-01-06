"""
Download Satellite Images and Run Solar Classification Pipeline
================================================================
Workflow:
  1. Read lat/lon from Excel file in input/ folder
  2. Download satellite images from Google Static Maps API
  3. Run solar_classify.py on each downloaded image
  4. Save classified outputs to artefacts/test/
"""

import os
import sys
import requests
from pathlib import Path
import pandas as pd

# Add pipeline_code to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from solar_classify import SolarPanelClassifier

# =============================================================
# CONFIGURATION
# =============================================================

# Paths - relative to project root
PROJECT_ROOT = script_dir.parent
INPUT_DIR = PROJECT_ROOT / "input"
DOWNLOAD_DIR = PROJECT_ROOT / "prediction_files" / "test"
OUTPUT_DIR = PROJECT_ROOT / "artefacts" / "test"

# Create directories
DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Google Static Maps API settings
API_KEY = "<Your api key>"  # Your API key

# Image settings
ZOOM = 20  # High zoom for rooftop detail (18-21)
SIZE = "640x640"  # Maximum free size
MAPTYPE = "satellite"
SCALE = 2  # 2x resolution (1280x1280 actual)


# =============================================================
# FUNCTIONS
# =============================================================

def find_excel_file():
    """Find the Excel file in the input folder."""
    excel_extensions = ['.xlsx', '.xls', '.csv']
    
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(exist_ok=True, parents=True)
        print(f"⚠️ Created input directory: {INPUT_DIR}")
        return None
    
    for ext in excel_extensions:
        files = list(INPUT_DIR.glob(f"*{ext}"))
        if files:
            return files[0]  # Return first matching file
    
    return None


def read_locations_from_excel(excel_path):
    """
    Read lat/lon coordinates from Excel file.
    
    Expected columns:
      - latitude (or lat)
      - longitude (or lon or lng)
      - name (optional, for naming the output files)
      - sample_id (optional, alternative to name)
    """
    print(f"📖 Reading locations from: {excel_path}")
    
    # Read based on file extension
    if str(excel_path).endswith('.csv'):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)
    
    print(f"   Found {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Normalize column names (lowercase, strip spaces)
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Find latitude column
    lat_col = None
    for col in ['latitude', 'lat']:
        if col in df.columns:
            lat_col = col
            break
    
    # Find longitude column
    lon_col = None
    for col in ['longitude', 'lon', 'lng']:
        if col in df.columns:
            lon_col = col
            break
    
    if lat_col is None or lon_col is None:
        print(f"❌ Error: Could not find latitude/longitude columns!")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Expected: 'latitude' or 'lat' AND 'longitude', 'lon', or 'lng'")
        return []
    
    # Find name column
    name_col = None
    for col in ['name', 'sample_id', 'id', 'location']:
        if col in df.columns:
            name_col = col
            break
    
    # Build locations list
    locations = []
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Skip invalid coordinates
        if pd.isna(lat) or pd.isna(lon):
            print(f"   ⚠️ Skipping row {idx+1}: missing coordinates")
            continue
        
        # Get name or generate one
        if name_col and pd.notna(row[name_col]):
            name = str(row[name_col]).replace(' ', '_')
        else:
            name = f"location_{idx+1}"
        
        locations.append({
            'name': name,
            'lat': float(lat),
            'lon': float(lon)
        })
    
    print(f"   ✅ Loaded {len(locations)} valid locations")
    return locations


def download_satellite_image(lat, lon, name, api_key):
    """Download a satellite image from Google Static Maps API."""
    
    url = f"https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": SIZE,
        "scale": SCALE,
        "maptype": MAPTYPE,
        "key": api_key
    }
    
    output_path = DOWNLOAD_DIR / f"{name}.png"
    
    print(f"   📡 Downloading: {name} ({lat}, {lon})...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"      ✅ Saved: {output_path.name}")
            return output_path
        else:
            print(f"      ❌ Failed: HTTP {response.status_code}")
            print(f"         Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def run_classification(image_path, classifier):
    """Run solar classification on an image and save to artefacts/test."""
    print(f"   🔍 Classifying: {image_path.name}...")
    
    try:
        result = classifier.classify(str(image_path))
        
        # The classifier saves output to test_output_images by default
        # Move/copy to artefacts/test
        if result.get('output_image'):
            src = Path(result['output_image'])
            if src.exists():
                import shutil
                dst = OUTPUT_DIR / src.name
                shutil.copy2(src, dst)
                print(f"      ✅ Output saved: {dst.name}")
                print(f"         PV Present: {result.get('pv_present', False)}")
                print(f"         Panel Area: {result.get('panel_area_m2', 0):.2f} m²")
                return result
        
        return result
        
    except Exception as e:
        print(f"      ❌ Classification error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_sample_excel():
    """Create a sample Excel file in the input folder."""
    sample_data = {
        'name': ['sample_location_1', 'sample_location_2'],
        'latitude': [12.971891, 13.082680],
        'longitude': [77.594624, 80.270718]
    }
    
    sample_path = INPUT_DIR / "input.xlsx"
    df = pd.DataFrame(sample_data)
    df.to_excel(sample_path, index=False)
    
    print(f"\n📝 Created sample Excel file: {sample_path}")
    print("   Please edit this file with your coordinates and run again.")
    return sample_path


def main():
    print("=" * 70)
    print("SOLAR PANEL DETECTION PIPELINE")
    print("Download Satellite Images → Classify → Save Results")
    print("=" * 70)
    print(f"Input folder:    {INPUT_DIR}")
    print(f"Download folder: {DOWNLOAD_DIR}")
    print(f"Output folder:   {OUTPUT_DIR}")
    print("=" * 70)
    
    # Step 1: Find and read Excel file
    excel_file = find_excel_file()
    
    if excel_file is None:
        print("\n⚠️ No Excel file found in input folder!")
        create_sample_excel()
        return
    
    locations = read_locations_from_excel(excel_file)
    
    if not locations:
        print("\n❌ No valid locations found in Excel file!")
        return
    
    print(f"\n📍 Processing {len(locations)} locations...")
    print("=" * 70)
    
    # Step 2: Initialize classifier (do this once to save time)
    print("\n🤖 Initializing Solar Panel Classifier...")
    classifier = SolarPanelClassifier()
    
    # Step 3: Process each location
    results = []
    
    for i, loc in enumerate(locations, 1):
        print(f"\n[{i}/{len(locations)}] {loc['name']}")
        print("-" * 50)
        
        # Download satellite image
        image_path = download_satellite_image(
            loc['lat'], loc['lon'], loc['name'], API_KEY
        )
        
        if image_path and image_path.exists():
            # Run classification
            result = run_classification(image_path, classifier)
            if result:
                result['location'] = loc
                results.append(result)
        else:
            print(f"   ⚠️ Skipping classification - image download failed")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total locations processed: {len(locations)}")
    print(f"Successful classifications: {len(results)}")
    print(f"\n📁 Downloaded images:     {DOWNLOAD_DIR}")
    print(f"📁 Classified outputs:    {OUTPUT_DIR}")
    
    # List output files
    output_files = list(OUTPUT_DIR.glob("*_classified.jpg"))
    print(f"\nOutput files ({len(output_files)}):")
    for f in output_files:
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()
