"""
Download Satellite Images from Google Static Maps API
======================================================
Downloads high-resolution satellite imagery for given lat/long coordinates.
"""

import os
import requests
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("F:/AURA/satellite_test_images")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Google Static Maps API settings
# NOTE: You need a valid API key for this to work
API_KEY = "your-API-key"  # Replace with your actual API key

# Image settings
ZOOM = 20  # High zoom for rooftop detail (18-21)
SIZE = "640x640"  # Maximum free size
MAPTYPE = "satellite"
SCALE = 2  # 2x resolution (1280x1280 actual)

# Locations to download
LOCATIONS = [
    {"name": "bengaluru_residential", "lat": 12.971891, "lon": 77.594624},
    {"name": "bengaluru_apartment_solar", "lat": 12.935242, "lon": 77.624481},
    {"name": "bengaluru_commercial", "lat": 12.998174, "lon": 77.695879},
    {"name": "chennai_rooftop_pv", "lat": 13.082680, "lon": 80.270718},
    {"name": "chennai_apartment", "lat": 13.034222, "lon": 80.230173},
    {"name": "hyderabad_dense_solar", "lat": 17.385044, "lon": 78.486671},
    {"name": "hyderabad_gated_community", "lat": 17.447412, "lon": 78.376229},
    {"name": "mumbai_industrial", "lat": 19.076090, "lon": 72.877426},
    {"name": "mumbai_mall_solar", "lat": 19.113645, "lon": 72.869733},
    {"name": "pune_residential", "lat": 18.520430, "lon": 73.856744},
    {"name": "pune_it_park", "lat": 18.560213, "lon": 73.776937},
    {"name": "delhi_government", "lat": 28.613939, "lon": 77.209023},
    {"name": "noida_commercial", "lat": 28.535517, "lon": 77.391029},
    {"name": "kolkata_warehouse", "lat": 22.572646, "lon": 88.363895},
    {"name": "kolkata_residential", "lat": 22.596721, "lon": 88.263639},
    {"name": "ahmedabad_factory", "lat": 23.022505, "lon": 72.571362},
    {"name": "ahmedabad_society", "lat": 23.030357, "lon": 72.517845},
    {"name": "jaipur_institutional", "lat": 26.912434, "lon": 75.787271},
    {"name": "lucknow_rooftop_pv", "lat": 26.846708, "lon": 80.946159},
    {"name": "coimbatore_industrial", "lat": 11.016844, "lon": 76.955833},
]


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
    
    output_path = OUTPUT_DIR / f"{name}_z{ZOOM}.png"
    
    print(f"Downloading: {name} ({lat}, {lon})...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✅ Saved: {output_path}")
            return True
        else:
            print(f"  ❌ Failed: HTTP {response.status_code}")
            print(f"     Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Google Static Maps Satellite Image Downloader")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Zoom level: {ZOOM}")
    print(f"Image size: {SIZE} (scale {SCALE}x = actual 1280x1280)")
    print(f"Total locations: {len(LOCATIONS)}")
    print("=" * 60)
    
    if API_KEY == "YOUR_GOOGLE_MAPS_API_KEY":
        print("\n⚠️  WARNING: You need to set a valid Google Maps API key!")
        print("   1. Go to: https://console.cloud.google.com/")
        print("   2. Enable 'Maps Static API'")
        print("   3. Create an API key")
        print("   4. Replace API_KEY in this script")
        print("\n   Alternatively, you can download images manually from Google Earth.")
        
        # Create a text file with all coordinates for manual download
        coords_file = OUTPUT_DIR / "coordinates.txt"
        with open(coords_file, 'w') as f:
            f.write("Satellite Image Coordinates\n")
            f.write("=" * 40 + "\n\n")
            for loc in LOCATIONS:
                f.write(f"{loc['name']}:\n")
                f.write(f"  Lat: {loc['lat']}\n")
                f.write(f"  Lon: {loc['lon']}\n")
                f.write(f"  Google Maps: https://www.google.com/maps/@{loc['lat']},{loc['lon']},{ZOOM}z/data=!3m1!1e3\n\n")
        print(f"\n📄 Coordinates saved to: {coords_file}")
        return
    
    success = 0
    failed = 0
    
    for loc in LOCATIONS:
        if download_satellite_image(loc['lat'], loc['lon'], loc['name'], API_KEY):
            success += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Download complete: {success} success, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
