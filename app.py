#!/usr/bin/env python3
"""
NRI Indicator Dashboard - Standalone Deployment Version
"""

import sys
import os

# Import the visualization function
from visualization_app import run_visualization_app

if __name__ == '__main__':
    # Use the data file in the same directory
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    
    # Get port from environment variable (for cloud deployment) or use default
    port = int(os.environ.get('PORT', 8050))
    
    # Run the app
    run_visualization_app(
        data_path=data_path,
        dashboard_title='NRI Indicator Dashboard',
        port=port,
        debug=False,  # Set to False for production
        openBrowser=False  # Don't auto-open browser on server
    )
