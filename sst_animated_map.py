#!/usr/bin/env python3
"""
Animated Sea Surface Temperature WebMap
Data source: NOAA ERSST v5 Monthly Mean
https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc
"""

import os
import requests
import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
DATA_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
DATA_FILE = "sst.mnmean.nc"
OUTPUT_FILE = "sst_animated_map.html"

# Subsampling factor for performance (1 = full resolution, 2 = half, etc.)
# Increase this if the animation is too slow
SPATIAL_SUBSAMPLE = 1
# Time step (1 = every month, 12 = yearly, etc.)
TIME_STEP = 1  # Monthly data (all 2064 frames)


def download_data():
    """Download the NetCDF file if not already present."""
    if os.path.exists(DATA_FILE):
        print(f"Data file '{DATA_FILE}' already exists, skipping download.")
        return

    print(f"Downloading data from {DATA_URL}...")
    print("This may take a few minutes (file is ~139MB)...")

    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(DATA_FILE, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                print(f"\rProgress: {pct:.1f}%", end="", flush=True)

    print(f"\nDownload complete: {DATA_FILE}")


def load_and_process_data():
    """Load and process the NetCDF data."""
    print("Loading NetCDF data...")
    ds = xr.open_dataset(DATA_FILE)

    print("\nDataset info:")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Total time steps: {len(ds.time)}")

    # Extract SST data
    sst = ds['sst']

    # Subsample spatially if needed
    if SPATIAL_SUBSAMPLE > 1:
        sst = sst.isel(
            lat=slice(None, None, SPATIAL_SUBSAMPLE),
            lon=slice(None, None, SPATIAL_SUBSAMPLE)
        )

    # Subsample temporally
    sst = sst.isel(time=slice(None, None, TIME_STEP))

    print(f"\nAfter subsampling:")
    print(f"  Shape: {sst.shape}")
    print(f"  Time steps for animation: {len(sst.time)}")

    return sst, ds


def create_animation(sst, ds):
    """Create the animated Plotly visualization."""
    print("\nCreating animation frames...")

    # Get coordinate arrays
    lons = sst.lon.values
    lats = sst.lat.values
    times = sst.time.values

    # Convert longitudes from 0-360 to -180 to 180 for better display
    lons_converted = np.where(lons > 180, lons - 360, lons)
    sort_idx = np.argsort(lons_converted)
    lons_sorted = lons_converted[sort_idx]

    # Prepare frames
    frames = []

    for i, t in enumerate(times):
        # Get SST data for this time step
        sst_data = sst.sel(time=t).values

        # Reorder longitudes
        sst_sorted = sst_data[:, sort_idx]

        # Convert time to string for display
        time_str = str(t)[:7]  # YYYY-MM format

        frame = go.Frame(
            data=[go.Heatmap(
                z=sst_sorted,
                x=lons_sorted,
                y=lats,
                colorscale='RdBu_r',
                zmin=-2,
                zmax=32,
                colorbar=dict(
                    title=dict(text='SST (°C)', side='right')
                ),
                hoverongaps=False,
                hovertemplate='Lon: %{x:.1f}°<br>Lat: %{y:.1f}°<br>SST: %{z:.2f}°C<extra></extra>'
            )],
            name=time_str,
            layout=go.Layout(title=f"Sea Surface Temperature - {time_str}")
        )
        frames.append(frame)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(times)} frames...")

    print(f"  Total frames created: {len(frames)}")

    # Create initial figure with first frame's data
    initial_sst = sst.isel(time=0).values[:, sort_idx]
    initial_time = str(times[0])[:7]

    fig = go.Figure(
        data=[go.Heatmap(
            z=initial_sst,
            x=lons_sorted,
            y=lats,
            colorscale='RdBu_r',
            zmin=-2,
            zmax=32,
            colorbar=dict(
                title=dict(text='SST (°C)', side='right'),
                len=0.75
            ),
            hoverongaps=False,
            hovertemplate='Lon: %{x:.1f}°<br>Lat: %{y:.1f}°<br>SST: %{z:.2f}°C<extra></extra>'
        )],
        frames=frames
    )

    # Create slider steps
    slider_steps = []
    for i, t in enumerate(times):
        time_str = str(t)[:7]
        step = dict(
            method='animate',
            args=[[time_str], dict(
                mode='immediate',
                frame=dict(duration=100, redraw=True),
                transition=dict(duration=0)
            )],
            label=time_str
        )
        slider_steps.append(step)

    # Update layout with animation controls
    fig.update_layout(
        title=dict(
            text=f"NOAA ERSST v5 - Sea Surface Temperature ({initial_time})",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Longitude',
            range=[-180, 180],
            dtick=60,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            title='Latitude',
            range=[-90, 90],
            dtick=30,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            scaleanchor='x',
            scaleratio=1
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.1,
                xanchor='right',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=14),
                prefix='Date: ',
                visible=True,
                xanchor='center'
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.05,
            y=0,
            steps=slider_steps
        )],
        margin=dict(l=60, r=60, t=100, b=80),
        paper_bgcolor='white',
        plot_bgcolor='lightgray'
    )

    # Add coastline approximation using a contour at 0
    # (This is approximate - land areas show as NaN/gray)

    return fig


def main():
    """Main function to run the animation creation."""
    print("=" * 60)
    print("NOAA ERSST v5 Animated Sea Surface Temperature Map")
    print("=" * 60)

    # Download data
    download_data()

    # Load and process data
    sst, ds = load_and_process_data()

    # Create animation
    fig = create_animation(sst, ds)

    # Save to HTML
    print(f"\nSaving animation to '{OUTPUT_FILE}'...")
    fig.write_html(
        OUTPUT_FILE,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False
    )

    print(f"\nDone! Open '{OUTPUT_FILE}' in your web browser to view the animation.")
    print("\nControls:")
    print("  - Click 'Play' to start the animation")
    print("  - Use the slider to navigate to specific dates")
    print("  - Hover over the map to see SST values")

    # Close dataset
    ds.close()


if __name__ == "__main__":
    main()
