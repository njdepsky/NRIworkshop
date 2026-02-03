#!/usr/bin/env python3
"""
NRI Indicator Dashboard - Cloud Deployment Version
"""

import sys
import os

# Import the visualization function
from visualization_app import run_visualization_app

# Try multiple paths to find data.csv
possible_paths = [
    'data.csv',  # Current directory
    os.path.join(os.path.dirname(__file__), 'data.csv'),  # Same dir as script
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv'),  # Absolute path
    os.path.join(os.getcwd(), 'data.csv'),  # Working directory
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        print(f"Found data file at: {path}")
        break

if data_path is None:
    print(f"ERROR: Could not find data.csv")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Files in current directory: {os.listdir('.')}")
    sys.exit(1)

# Try to find GeoJSON file
geojson_paths = [
    'geom.geojson',
    os.path.join(os.path.dirname(__file__), 'geom.geojson'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geom.geojson'),
    os.path.join(os.getcwd(), 'geom.geojson'),
]

geojson_path = None
for path in geojson_paths:
    if os.path.exists(path):
        geojson_path = path
        print(f"Found GeoJSON file at: {path}")
        break

if geojson_path is None:
    print("Warning: No GeoJSON file found, will try to use geometry column in CSV")


# Get port from environment variable (for cloud deployment) or use default
port = int(os.environ.get('PORT', 8050))

# Run the app
run_visualization_app(
    data_path=data_path,
    geojson_path=geojson_path,  # Pass GeoJSON path (or None to use CSV geometry)
    dashboard_title='NRI Indicator Dashboard',
    port=port,
    debug=False,  # Set to False for production
    openBrowser=False  # Don't auto-open browser on server
)
