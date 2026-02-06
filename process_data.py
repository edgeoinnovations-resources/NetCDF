#!/usr/bin/env python3
"""
Convert NetCDF SST data to compact binary format for web visualization.

Input:  sst.mnmean.nc (NOAA ERSST v5 Monthly Mean)
Output: docs/data/metadata.json  - grid info, time labels, bounds
        docs/data/sst_data.bin   - uint8 binary array [time × lat × lon]

Encoding: SST values mapped from [-2°C, 32°C] → [0, 254] uint8
          255 = no data (NaN / land)
          Resolution: 34°C / 254 ≈ 0.134°C per step
"""

import os
import json
import numpy as np
import xarray as xr

DATA_FILE = "sst.mnmean.nc"
OUTPUT_DIR = "docs/data"

# SST mapping range
SST_MIN = -2.0
SST_MAX = 32.0
SST_RANGE = SST_MAX - SST_MIN
NODATA = 255


def process():
    """Convert NetCDF to binary format for web visualization."""
    print("=" * 60)
    print("NetCDF → Binary Data Processor")
    print("=" * 60)

    print(f"\nLoading {DATA_FILE}...")
    ds = xr.open_dataset(DATA_FILE)
    sst = ds['sst']

    lons = sst.lon.values
    lats = sst.lat.values
    times = sst.time.values

    print(f"  Grid: {len(lats)} lat × {len(lons)} lon")
    print(f"  Time range: {str(times[0])[:10]} → {str(times[-1])[:10]}")
    print(f"  Time steps: {len(times)}")

    # Convert longitudes from 0-360 to -180 to 180
    lons_converted = np.where(lons > 180, lons - 360, lons)
    lon_sort = np.argsort(lons_converted)
    lons_sorted = lons_converted[lon_sort]

    # Flip latitudes to go north → south (image convention: top = north)
    lat_flip = np.argsort(-lats)
    lats_sorted = lats[lat_flip]

    nlat = len(lats)
    nlon = len(lons)
    ntimes = len(times)

    # Time labels in YYYY-MM format
    time_labels = [str(t)[:7] for t in times]

    # Half cell size for bounds (data points are cell centers)
    half_lon = float(abs(lons_sorted[1] - lons_sorted[0]) / 2)
    half_lat = float(abs(lats_sorted[1] - lats_sorted[0]) / 2)

    # Build metadata
    metadata = {
        "nlat": nlat,
        "nlon": nlon,
        "ntimes": ntimes,
        "times": time_labels,
        "lats": [float(x) for x in lats_sorted],
        "lons": [float(x) for x in lons_sorted],
        "bounds": {
            "north": float(lats_sorted[0] + half_lat),
            "south": float(lats_sorted[-1] - half_lat),
            "west": float(lons_sorted[0] - half_lon),
            "east": float(lons_sorted[-1] + half_lon)
        },
        "encoding": {
            "min": SST_MIN,
            "max": SST_MAX,
            "nodata": NODATA,
            "description": "uint8: 0-254 maps to -2°C to 32°C, 255 = no data"
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata.json")

    # Process all frames into a single binary array
    print(f"\nConverting {ntimes} time steps to uint8 binary...")
    binary_data = np.empty((ntimes, nlat, nlon), dtype=np.uint8)

    for i in range(ntimes):
        frame = sst.isel(time=i).values

        # Reorder: sort longitudes and flip latitudes
        frame = frame[:, lon_sort]
        frame = frame[lat_flip, :]

        # Map to uint8: [-2, 32] → [0, 254]
        normalized = (frame - SST_MIN) / SST_RANGE
        scaled = np.clip(normalized * 254, 0, 254).astype(np.uint8)
        scaled[np.isnan(frame)] = NODATA

        binary_data[i] = scaled

        if (i + 1) % 200 == 0 or i == 0:
            print(f"  Processed {i + 1}/{ntimes} frames...")

    # Save as raw binary
    output_path = f"{OUTPUT_DIR}/sst_data.bin"
    binary_data.tofile(output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    meta_size_kb = os.path.getsize(f"{OUTPUT_DIR}/metadata.json") / 1024

    print(f"\n{'=' * 60}")
    print(f"✓ Done!")
    print(f"  Binary data:  {file_size_mb:.1f} MB  ({output_path})")
    print(f"  Metadata:     {meta_size_kb:.1f} KB  ({OUTPUT_DIR}/metadata.json)")
    print(f"  Grid:         {nlat} × {nlon} = {nlat * nlon:,} cells/frame")
    print(f"  Frames:       {ntimes}")
    print(f"  Total cells:  {nlat * nlon * ntimes:,}")
    print(f"{'=' * 60}")

    ds.close()


if __name__ == "__main__":
    process()
