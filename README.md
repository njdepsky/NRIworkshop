# NRI Indicator Dashboard

Interactive data visualization dashboard for NRI (Natural Resource Index) indicators with choropleth maps and scatter plots.

## Features

- **Interactive Choropleth Map**: Visualize country-level indicator data on a world map
- **Multi-variable Scatter Plots**: Plot relationships between indicators with polynomial trendlines
- **Dynamic Histograms**: View distributions of selected variables
- **Data Transformations**: Log and normalization transforms
- **R² Statistics**: View model fit statistics for trendlines
- **Responsive Design**: Optimized for various screen sizes

## Quick Start

### Local Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

3. Open browser to: `http://localhost:8050`

## Files

- `app.py` - Main application entry point
- `visualization_app.py` - Dashboard implementation
- `data.csv` - NRI indicator data with geometry
- `requirements.txt` - Python dependencies

## Cloud Deployment (Free Options)

### Option 1: Render.com (Recommended)

1. Create account at https://render.com
2. Create new "Web Service"
3. Connect GitHub repository or upload files
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Environment: Python 3.11
5. Deploy (takes 5-10 minutes)

**Your app will be live at**: `https://your-app-name.onrender.com`

### Option 2: Railway.app

1. Create account at https://railway.app
2. Create new project
3. Deploy from GitHub or upload files
4. Set start command: `python app.py` in settings
5. Railway provides a public URL automatically

### Option 3: Fly.io

1. Install flyctl CLI
2. Run `fly launch`
3. Follow prompts
4. Deploy with `fly deploy`

## Configuration

Edit `app.py` to customize dashboard settings.

## Data Format

CSV file requires:
- Country columns (ADM0, COUNTRY)
- Numeric indicator columns
- `geometry` column with WKT MULTIPOLYGON data

## License

MIT License
