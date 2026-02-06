#!/usr/bin/env python3
"""
Convert NetCDF SST data to JSON format for web visualization.
Outputs compressed JSON files for efficient web loading.
"""

import os
import json
import gzip
import xarray as xr
import numpy as np

DATA_FILE = "sst.mnmean.nc"
OUTPUT_DIR = "public/data"

# Time step (1 = monthly, 12 = yearly)
TIME_STEP = 1


def convert_data():
    """Convert NetCDF to JSON format."""
    print("Loading NetCDF data...")
    ds = xr.open_dataset(DATA_FILE)
    sst = ds['sst']

    # Subsample temporally
    sst = sst.isel(time=slice(None, None, TIME_STEP))

    # Get coordinates
    lons = sst.lon.values.tolist()
    lats = sst.lat.values.tolist()
    times = sst.time.values

    # Convert longitudes from 0-360 to -180 to 180
    lons_converted = [lon - 360 if lon > 180 else lon for lon in lons]
    sort_idx = np.argsort(lons_converted)
    lons_sorted = [lons_converted[i] for i in sort_idx]

    print(f"Grid: {len(lats)} lat x {len(lons)} lon")
    print(f"Time steps: {len(times)}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save metadata
    time_labels = [str(t)[:7] for t in times]  # YYYY-MM format
    metadata = {
        "lons": lons_sorted,
        "lats": lats,
        "times": time_labels,
        "bounds": {
            "minLon": min(lons_sorted),
            "maxLon": max(lons_sorted),
            "minLat": min(lats),
            "maxLat": max(lats)
        },
        "valueRange": {"min": -2, "max": 32}
    }

    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f)
    print(f"Saved metadata.json")

    # Save SST data - one file per time step for efficient loading
    # Using integer encoding to reduce file size (value * 100, stored as int16)
    print("\nConverting SST data...")

    all_data = []
    for i, t in enumerate(times):
        sst_data = sst.sel(time=t).values
        # Reorder longitudes
        sst_sorted = sst_data[:, sort_idx]
        # Replace NaN with a sentinel value (-999) and scale
        sst_int = np.where(np.isnan(sst_sorted), -999, np.round(sst_sorted * 100)).astype(int)
        all_data.append(sst_int.tolist())

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(times)} time steps...")

    # Save as single compressed JSON file
    print("\nSaving compressed data file...")
    data_obj = {"data": all_data}

    # Save gzipped JSON
    with gzip.open(f"{OUTPUT_DIR}/sst_data.json.gz", 'wt', encoding='utf-8') as f:
        json.dump(data_obj, f, separators=(',', ':'))

    # Also save uncompressed for Firebase (it handles compression)
    with open(f"{OUTPUT_DIR}/sst_data.json", 'w') as f:
        json.dump(data_obj, f, separators=(',', ':'))

    # Get file sizes
    gz_size = os.path.getsize(f"{OUTPUT_DIR}/sst_data.json.gz") / (1024 * 1024)
    json_size = os.path.getsize(f"{OUTPUT_DIR}/sst_data.json") / (1024 * 1024)

    print(f"\nDone!")
    print(f"  Uncompressed: {json_size:.1f} MB")
    print(f"  Compressed:   {gz_size:.1f} MB")

    ds.close()


if __name__ == "__main__":
    convert_data()
