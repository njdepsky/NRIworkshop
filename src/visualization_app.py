import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import webbrowser
from threading import Timer


def run_visualization_app(
    data_path,
    # Scatter plot styling
    scatter_marker_color='#f0a179',
    scatter_marker_size=14,  # Updated from 7 to 9
    scatter_marker_opacity=0.75,
    scatter_marker_line_color='#b84515',
    scatter_marker_line_width=1,
    scatter_trendline_color='#4f4f4f',
    scatter_trendline_width=2,
    scatter_trendline_dash='dot',
    scatter_default_fit_degree=1,  # None = Auto (best model), or 1, 2, 3, 4
    scatter_default_y_variable=None,  # None = use second numeric column (or first if only one)
    # Scatter plot labels
    scatter_plot_title=None,  # None = auto-generate from vars
    scatter_title_fontsize=20,  # Unchanged
    scatter_title_bold=False,
    scatter_xlab=None,  # None = use variable name
    scatter_ylab=None,  # None = use variable name
    scatter_axis_label_fontsize=24,  # Updated from 12 to 18
    scatter_axis_label_bold=True,  # Updated from False to True
    scatter_tick_fontsize=20,  # Updated from 10 to 16
    # Dashboard title
    dashboard_title='NRI Indicator Dashboard',
    dashboard_title_fontsize=28,
    # Font family
    font_family='Georgia',
    # Multiple Y variable colors (used when multiple Y vars selected)
    multi_y_colors=['#f0a179', '#7fb8d4', '#90c290', '#e8a0bf', '#f4d06f', '#b19cd9'],
    # Histogram styling
    histogram_colors=[('#7fb8d4', '#4a90b8'), ('#f0a179', '#b84515')],  # [(fill, edge), (fill, edge)]
    histogram_default_nbins=15,  # New: default number of bins
    histogram_xlab=None,  # New: None = use variable name
    histogram_ylab='Count',  # New: y-axis label
    histogram_label_fontsize=18,  # New: axis label font size
    histogram_tick_fontsize=16,  # New: tick label font size
    # Map styling
    map_default_colormap='YlGnBu',
    map_default_variable=None,  # None = use first numeric column
    map_default_vmin=None,
    map_default_vmax=None,
    map_initial_zoom=1.05,
    map_center_lat=10,
    map_center_lon=0,
    map_colorbar_title=False,
    map_colorbar_title_fontsize=18,
    map_colorbar_tick_fontsize=16,
    # Server settings
    port=8050,
    debug=True,
    openBrowser=True
):
    """
    Run interactive visualization dashboard
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with data and geometry column
    
    Scatter plot styling:
    scatter_marker_color : str
        Marker fill color (default: '#f0a179')
    scatter_marker_size : int
        Marker size (default: 9)
    scatter_marker_opacity : float
        Marker opacity 0-1 (default: 0.75)
    scatter_marker_line_color : str
        Marker edge color (default: '#b84515')
    scatter_marker_line_width : int
        Marker edge width (default: 1)
    scatter_trendline_color : str
        Trendline color (default: '#4f4f4f')
    scatter_trendline_width : int
        Trendline width (default: 2)
    scatter_trendline_dash : str
        Trendline style: 'solid', 'dot', 'dash' (default: 'dot')
    scatter_default_fit_degree : int or None
        Default polynomial degree: None = Auto (best model), 1, 2, 3, or 4 (default: None)
    scatter_default_y_variable : str or None
        Y variable to display at launch, None = second numeric column (default: None)
    
    Scatter plot labels:
    scatter_plot_title : str or None
        Plot title, None = auto-generate (default: None)
    scatter_title_fontsize : int
        Title font size (default: 12)
    scatter_title_bold : bool
        Bold title (default: False)
    scatter_xlab : str or None
        X-axis label, None = use variable name (default: None)
    scatter_ylab : str or None
        Y-axis label, None = use variable name (default: None)
    scatter_axis_label_fontsize : int
        Axis label font size (default: 18)
    scatter_axis_label_bold : bool
        Bold axis labels (default: True)
    scatter_tick_fontsize : int
        Tick label font size (default: 16)
    
    Dashboard:
    dashboard_title : str
        Main dashboard title (default: 'Interactive Data Visualization Dashboard')
    dashboard_title_fontsize : int
        Dashboard title font size (default: 24)
    
    Font:
    font_family : str
        Font family for all text (default: 'Arial')
    
    Histograms:
    histogram_colors : list of tuples
        [(x_fill, x_edge), (y_fill, y_edge)] (default: [('#7fb8d4', '#4a90b8'), ('#f0a179', '#b84515')])
    histogram_xlab : str or None
        X-axis label, None = use variable name (default: None)
    histogram_ylab : str
        Y-axis label (default: 'Count')
    histogram_label_fontsize : int
        Axis label font size (default: 14)
    histogram_tick_fontsize : int
        Tick label font size (default: 12)
    histogram_default_nbins : int
        Default number of bins (default: 15)
    histogram_xlab : str or None
        X-axis label, None = use variable name (default: None)
    histogram_ylab : str
        Y-axis label (default: 'Count')
    histogram_label_fontsize : int
        Axis label font size (default: 14)
    histogram_tick_fontsize : int
        Tick label font size (default: 12)
    
    Map styling:
    map_default_colormap : str
        Default colormap (default: 'YlGnBu')
    map_default_variable : str or None
        Variable to display on map at launch, None = first numeric column (default: None)
    map_default_vmin : float or None
        Default min value, None = auto (default: None)
    map_default_vmax : float or None
        Default max value, None = auto (default: None)
    map_initial_zoom : float
        Initial zoom level (default: 1.05)
    map_center_lat : float
        Center latitude (default: 10)
    map_center_lon : float
        Center longitude (default: 0)
    map_colorbar_title_fontsize : int
        Colorbar title font size (default: 12)
    map_colorbar_tick_fontsize : int
        Colorbar tick font size (default: 10)
    
    Server:
    port : int
        Server port (default: 8050)
    debug : bool
        Debug mode (default: True)
    """
    
    # Helper functions
    def weighted_r2(y, y_pred, w):
        """Calculate weighted R-squared"""
        if np.allclose(y, y.mean()) or np.isclose(np.sum(w), 0):
            return np.nan
        ybar = np.sum(w * y) / np.sum(w)
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - ybar) ** 2)
        return 1 - ss_res / ss_tot

    def fit_poly_choose_degree(x, y, fit_degree=None, fit_wts=None, want_pval=True):
        """Fit polynomial and choose best degree based on R-squared"""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if fit_wts is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(fit_wts, float)
            w[w < 0] = 0.0

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        x, y, w = x[m], y[m], w[m]
        sqrt_w = np.sqrt(w)

        def fit_one_deg(deg):
            coeffs = np.polyfit(x, y, deg=deg, w=sqrt_w if fit_wts is not None else None)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)
            r2w = weighted_r2(y, y_pred, w if fit_wts is not None else np.ones_like(y))
            return poly, r2w, coeffs

        degrees = [fit_degree] if fit_degree is not None else [1, 2, 3]

        best_model, best_r2, best_degree, best_pval = None, -np.inf, None, None
        best_slope, best_intercept = None, None

        for deg in degrees:
            poly, r2w, coeffs = fit_one_deg(deg)
            if r2w > best_r2:
                best_model, best_r2, best_degree = poly, r2w, deg

                if deg == 1:
                    best_slope, best_intercept = coeffs
                    if want_pval:
                        X = sm.add_constant(x)
                        model = sm.WLS(
                            y, X, weights=w if fit_wts is not None else np.ones_like(y)
                        ).fit()
                        best_pval = float(model.pvalues[1])
                    else:
                        best_pval = None
                else:
                    best_pval, best_slope, best_intercept = None, None, None

        best_r2_display = max(0.0, best_r2) if np.isfinite(best_r2) else np.nan

        return {
            "model": best_model,
            "degree": best_degree,
            "r2": best_r2,
            "r2_display": best_r2_display,
            "pval": best_pval,
            "slope": best_slope,
            "intercept": best_intercept,
        }

    def normMinMax(arr):
        """Normalize array to 0-1 range"""
        arr = np.asarray(arr, float)
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val == min_val:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def transform_data(data, apply_log=False, apply_norm=False):
        """Apply log transform and/or normalization to data"""
        data = np.asarray(data, float).copy()
        
        if apply_log:
            data = np.log10(data + 1e-10)
        
        if apply_norm:
            data = normMinMax(data)
        
        return data

    def create_scatter_plot(df, x_col, y_cols, hover_cols=None, log_x=False, log_y=False, 
                           norm_x=False, norm_y=False, fit_degree=None):
        """Create a Plotly scatter plot with trendline - supports multiple Y variables"""
        if not isinstance(y_cols, list):
            y_cols = [y_cols]
        
        x = df[x_col].values
        x_transformed = transform_data(x, apply_log=log_x, apply_norm=norm_x)
        
        x_label = scatter_xlab if scatter_xlab else x_col
        if log_x:
            x_label = f"Log({x_label})"
        if norm_x:
            x_label = f"{x_label} (normalized)"
        
        fig = go.Figure()
        r2_values = []  # Store R² for each Y variable
        
        # Plot each Y variable as a separate series
        for idx, y_col in enumerate(y_cols):
            y = df[y_col].values
            y_transformed = transform_data(y, apply_log=log_y, apply_norm=norm_y)
            
            y_label = scatter_ylab if scatter_ylab else y_col
            if log_y:
                y_label = f"Log({y_label})"
            if norm_y:
                y_label = f"{y_label} (normalized)"
            
            keep_idx = np.isfinite(x_transformed) & np.isfinite(y_transformed)
            x_clean = x_transformed[keep_idx]
            y_clean = y_transformed[keep_idx]
            
            # Get color for this series and corresponding darker edge
            color = multi_y_colors[idx % len(multi_y_colors)]
            
            # Create darker edge color for visibility
            if color == '#f0a179':  # Orange
                edge_color = '#b84515'
            elif color == '#7fb8d4':  # Blue
                edge_color = '#4a90b8'
            elif color == '#90c290':  # Green
                edge_color = '#5a8a5a'
            elif color == '#e8a0bf':  # Pink
                edge_color = '#b8708f'
            elif color == '#f4d06f':  # Yellow
                edge_color = '#c4a03f'
            elif color == '#b19cd9':  # Purple
                edge_color = '#816ca9'
            else:
                edge_color = '#333333'  # Default dark edge
            
            hover_text = []
            for i in range(len(x_clean)):
                text_parts = [
                    f"<b>{x_label}:</b> {x_clean[i]:.4g}",
                    f"<b>{y_label}:</b> {y_clean[i]:.4g}"
                ]
                
                if hover_cols:
                    for col in hover_cols:
                        if col in df.columns:
                            value = df[col].values[keep_idx][i]
                            if isinstance(value, (int, np.integer)):
                                text_parts.append(f"<b>{col}:</b> {value}")
                            elif isinstance(value, (float, np.floating)):
                                text_parts.append(f"<b>{col}:</b> {value:.4g}")
                            else:
                                text_parts.append(f"<b>{col}:</b> {value}")
                
                hover_text.append("<br>".join(text_parts))
            
            # Add scatter trace
            fig.add_trace(go.Scatter(
                x=x_clean,
                y=y_clean,
                mode='markers',
                name=y_col,  # Use variable name for legend
                marker=dict(
                    size=scatter_marker_size,
                    color=color,
                    opacity=scatter_marker_opacity,
                    line=dict(color=edge_color, width=scatter_marker_line_width)  # Darker edge
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=len(y_cols) > 1  # Only show legend if multiple Y vars
            ))
            
            # Add trendline for this series
            best_fit = fit_poly_choose_degree(x_clean, y_clean, fit_degree=fit_degree, fit_wts=None)
            best_model = best_fit["model"]
            
            # Store R² and color for annotation
            r2_values.append({
                'y_var': y_col,
                'r2': best_fit["r2_display"],
                'color': color
            })
            
            if best_model is not None:
                x_sorted = np.linspace(min(x_clean), max(x_clean), 500)
                y_fit = best_model(x_sorted)
                
                fig.add_trace(go.Scatter(
                    x=x_sorted,
                    y=y_fit,
                    mode='lines',
                    name=f'{y_col} fit',
                    line=dict(color=color, width=scatter_trendline_width, dash=scatter_trendline_dash),
                    opacity=0.8,
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Create title
        if len(y_cols) == 1:
            y_label_title = scatter_ylab if scatter_ylab else y_cols[0]
            if log_y:
                y_label_title = f"Log({y_label_title})"
            if norm_y:
                y_label_title = f"{y_label_title} (normalized)"
            title_text = scatter_plot_title 
        else:
            title_text = scatter_plot_title
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(family=font_family, size=scatter_title_fontsize, color='#4f4f4f'),
                x=0.02,
                xanchor='left',
                pad=dict(t=10, b=10)
            ),
            xaxis=dict(
                title=dict(text=x_label, font=dict(family=font_family, size=scatter_axis_label_fontsize, color='#4f4f4f')),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                griddash='dash',
                tickfont=dict(family=font_family, size=scatter_tick_fontsize, color='#4f4f4f')
            ),
            yaxis=dict(
                title=dict(text='Value' if len(y_cols) > 1 else y_label, 
                          font=dict(family=font_family, size=scatter_axis_label_fontsize, color='#4f4f4f')),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                griddash='dash',
                tickfont=dict(family=font_family, size=scatter_tick_fontsize, color='#4f4f4f')
            ),
            template='plotly_white',
            hovermode='closest',
            showlegend=len(y_cols) > 1,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0,
                font=dict(family=font_family, size=16)
            ) if len(y_cols) > 1 else None,
            font=dict(family=font_family),
            margin=dict(l=60, r=20, t=90 if len(y_cols) > 1 else 70, b=60),
            autosize=True,
            height=None
        )
        
        return fig, r2_values

    def create_histogram(df, col, color=None, apply_log=False, apply_norm=False, nbins=15):
        """Create a histogram for a single column with optional custom color"""
        # Use provided color or default from histogram_colors
        if color is None:
            fill_color = '#cccccc'  # Light gray default
            edge_color = '#666666'  # Dark gray edge
        else:
            fill_color = color
            # Create darker edge for visibility
            if color == '#cccccc':  # X variable (light gray)
                edge_color = '#666666'  # Dark gray edge
            elif color == '#f0a179':  # Orange
                edge_color = '#b84515'  # Dark orange edge
            elif color == '#7fb8d4':  # Blue
                edge_color = '#4a90b8'  # Dark blue edge
            elif color == '#90c290':  # Green
                edge_color = '#5a8a5a'  # Dark green edge
            elif color == '#e8a0bf':  # Pink
                edge_color = '#b8708f'  # Dark pink edge
            elif color == '#f4d06f':  # Yellow
                edge_color = '#c4a03f'  # Dark yellow edge
            elif color == '#b19cd9':  # Purple
                edge_color = '#816ca9'  # Dark purple edge
            else:
                edge_color = '#333333'  # Default dark edge
        
        data = df[col].dropna().values
        data_transformed = transform_data(data, apply_log=apply_log, apply_norm=apply_norm)
        
        # Count non-NaN values
        n_count = len(data)
        
        # Use custom x-label if provided, otherwise use variable name
        if histogram_xlab is not None:
            label = histogram_xlab
        else:
            label = col
            if apply_log:
                label = f"Log({label})"
            if apply_norm:
                label = f"{label} (normalized)"
        
        fig = go.Figure()
        
        data_min, data_max = np.nanmin(data_transformed), np.nanmax(data_transformed)
        data_range = data_max - data_min
        bin_size = data_range / nbins * 1.0001
        
        fig.add_trace(go.Histogram(
            x=data_transformed,
            xbins=dict(
                start=data_min,
                end=data_min + bin_size * nbins,
                size=bin_size
            ),
            autobinx=False,
            marker=dict(
                color=fill_color,
                line=dict(color=edge_color, width=1.5)  # Thicker border for visibility
            ),
            opacity=0.75,
            name=label
        ))
        
        # Create y-axis label with count
        ylab_with_count = f"{histogram_ylab} (n = {n_count})"
        
        fig.update_layout(
            xaxis=dict(
                title=dict(text=label, font=dict(family=font_family, size=histogram_label_fontsize, color='#4f4f4f')),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                griddash='dash',
                tickfont=dict(family=font_family, size=histogram_tick_fontsize, color='#4f4f4f')
            ),
            yaxis=dict(
                title=dict(text=ylab_with_count, font=dict(family=font_family, size=histogram_label_fontsize, color='#4f4f4f')),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                griddash='dash',
                tickfont=dict(family=font_family, size=histogram_tick_fontsize, color='#4f4f4f')
            ),
            template='plotly_white',
            showlegend=False,
            font=dict(family=font_family),
            margin=dict(l=60, r=20, t=20, b=50),
            autosize=True,
            height=None
        )
        
        return fig

    def create_choropleth_map(df, value_col, apply_log=False, apply_norm=False, 
                             colormap=None, vmin=None, vmax=None):
        """Create a choropleth map using country geometry"""
        if colormap is None:
            colormap = map_default_colormap
            
        values = df[value_col].values
        values_transformed = transform_data(values, apply_log=apply_log, apply_norm=apply_norm)
        
        # Count non-NaN finite values for display
        n_count = np.sum(np.isfinite(values_transformed))
        
        if value_col == '_temp_avg':
            label = "Average"
        else:
            label = value_col
        
        if apply_log:
            label = f"Log({label})"
        if apply_norm:
            label = f"{label} (normalized)"
        
        # Add count to label with extra spacing
        label_with_count = f"<br>n = {n_count}<br>{label}"
        
        finite_values = values_transformed[np.isfinite(values_transformed)]
        if len(finite_values) > 0:
            actual_vmin = np.nanmin(finite_values) if vmin is None else vmin
            actual_vmax = np.nanmax(finite_values) if vmax is None else vmax
        else:
            actual_vmin = map_default_vmin if map_default_vmin is not None else 0
            actual_vmax = map_default_vmax if map_default_vmax is not None else 1
        
        if 'geometry' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No geometry data available.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#999')
            )
            fig.update_layout(height=650, template='plotly_white', margin=dict(l=0, r=150, t=70, b=0))
            return fig, actual_vmin, actual_vmax
        
        features_with_data = []
        features_no_data = []
        
        for idx, row in df.iterrows():
            if idx in parsed_geometries:
                geom_json = parsed_geometries[idx]
                
                has_data = pd.notna(values_transformed[idx]) and np.isfinite(values_transformed[idx])
                
                feature = {
                    "type": "Feature",
                    "id": str(idx),
                    "properties": {
                        "value": values_transformed[idx] if has_data else None,
                        "country": row.get('COUNTRY', row.get('ADM0', f'Country_{idx}')),
                        "label": label
                    },
                    "geometry": geom_json
                }
                
                if has_data:
                    features_with_data.append(feature)
                else:
                    features_no_data.append(feature)
        
        if not features_with_data and not features_no_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid geometry data found.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='#999')
            )
            fig.update_layout(height=650, template='plotly_white', margin=dict(l=0, r=150, t=70, b=0))
            return fig, actual_vmin, actual_vmax
        
        geojson_with_data = {"type": "FeatureCollection", "features": features_with_data}
        geojson_no_data = {"type": "FeatureCollection", "features": features_no_data}
        
        fig = go.Figure()
        
        if features_no_data:
            locations_no_data = [f["id"] for f in features_no_data]
            z_no_data = [0] * len(features_no_data)
            hover_text_no_data = [f["properties"]["country"] for f in features_no_data]
            
            fig.add_trace(go.Choropleth(
                geojson=geojson_no_data,
                locations=locations_no_data,
                z=z_no_data,
                text=hover_text_no_data,
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                showscale=False,
                marker_line_color='lightgrey',
                marker_line_width=0.5,
                hovertemplate='<b>%{text}</b><br>No data<extra></extra>'
            ))
        
        if features_with_data:
            locations = [f["id"] for f in features_with_data]
            z_values = [f["properties"]["value"] for f in features_with_data]
            hover_text = [f["properties"]["country"] for f in features_with_data]
            
            if not map_colorbar_title:
                label_with_count = f"n = {n_count}"
                
            fig.add_trace(go.Choropleth(
                geojson=geojson_with_data,
                locations=locations,
                z=z_values,
                text=hover_text,
                colorscale=colormap,
                zmin=actual_vmin,
                zmax=actual_vmax,
                marker_line_color='white',
                marker_line_width=0.5,
                colorbar=dict(
                    title=dict(
                        text=label_with_count,
                        font=dict(size=map_colorbar_title_fontsize, family=font_family)
                    ),
                    thickness=20,
                    len=0.7,
                    x=1.02,
                    xanchor='left',
                    tickfont=dict(size=map_colorbar_tick_fontsize, family=font_family)
                ),
                hovertemplate='<b>%{text}</b><br>' + label + ': %{z:.3f}<extra></extra>'
            ))
        
        fig.update_geos(
            visible=False,
            showcountries=False,
            showcoastlines=False,
            showland=False,
            projection_type="natural earth",
            projection_scale=map_initial_zoom,
            center=dict(lat=map_center_lat, lon=map_center_lon),
            lataxis_range=[-60, 85],
            lonaxis_range=[-180, 180]
        )
        
        fig.update_layout(
            height=None,
            margin=dict(l=0, r=150, t=0, b=0),
            font=dict(family=font_family),
            autosize=True,
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=False
        )
        
        return fig, actual_vmin, actual_vmax

    # Load data
    df = pd.read_csv(data_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Loaded {len(df)} countries with {len(numeric_cols)} numeric variables")
    
    # Pre-parse geometry
    print("Pre-parsing geometry data...")
    from shapely import wkt
    from shapely.geometry import mapping
    
    parsed_geometries = {}
    for idx, row in df.iterrows():
        if pd.notna(row.get('geometry')):
            try:
                geom = wkt.loads(row['geometry'])
                geom_json = mapping(geom)
                parsed_geometries[idx] = geom_json
            except Exception as e:
                print(f"Warning: Could not parse geometry for {row.get('COUNTRY', idx)}: {e}")
    
    print(f"Successfully parsed {len(parsed_geometries)} country geometries")
    
    # Initialize app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Add custom CSS for tab styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .custom-tab {
                    border: 1px solid #d3d3d3 !important;
                    border-bottom: none !important;
                    background-color: #e8e8e8 !important;
                    padding: 6px 12px !important;
                    font-family: Arial !important;
                    font-size: 14px !important;
                    font-weight: normal !important;
                    color: #333 !important;
                    text-align: center !important;
                }
                .custom-tab:hover {
                    background-color: #f5f5f5 !important;
                }
                .custom-tab--selected {
                    border: 1px solid #d3d3d3 !important;
                    border-bottom: 1px solid white !important;
                    background-color: white !important;
                    padding: 6px 12px !important;
                    font-family: Arial !important;
                    font-size: 14px !important;
                    font-weight: normal !important;
                    color: #333 !important;
                    text-align: center !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Layout
    app.layout = dbc.Container([
        # Title and tabs in same row
        html.Div([
            html.Div([
                html.H1(dashboard_title, 
                        style={'textAlign': 'left', 'margin': 0, 
                               'fontSize': f'{dashboard_title_fontsize}px', 'fontFamily': font_family})
            ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
            
            html.Div([
                dcc.Tabs(
                    id='view-tabs', 
                    value='map',
                    children=[
                        dcc.Tab(
                            label='Map', 
                            value='map',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            label='Plots', 
                            value='scatter',
                            className='custom-tab',
                            selected_className='custom-tab--selected'
                        ),
                    ],
                    style={'height': '36px'}
                )
            ], style={
                'display': 'inline-block',
                'float': 'right',
                'verticalAlign': 'middle',
                'width': '20%',
                'maxWidth': '300px',
                'border': '1px solid #d3d3d3', 
                'borderRadius': '4px',
                'padding': '5px',
                'backgroundColor': '#fafafa'
            }),
        ], id='title-div', style={'marginTop': '5px', 'marginBottom': '5px', 'overflow': 'hidden', 'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '4px'}),
        
        # Scatter controls
        html.Div(id='scatter-controls', children=[
            dbc.Row([
                dbc.Col([
                    html.Label("X variable:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': col, 'value': col} for col in numeric_cols],
                        value=numeric_cols[0] if numeric_cols else None,
                        clearable=False
                    ),
                ], width=3),  # Increased from 2 to 3
                dbc.Col([
                    html.Label("Y variable(s):", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    html.Div([
                        dcc.Checklist(
                            id='y-axis-checklist',
                            options=[{'label': col, 'value': col} for col in numeric_cols],
                            value=[scatter_default_y_variable] if scatter_default_y_variable and scatter_default_y_variable in numeric_cols else ([numeric_cols[1]] if len(numeric_cols) > 1 else [numeric_cols[0]]),
                            style={
                                'maxHeight': '120px', 
                                'overflowY': 'scroll',
                                'fontSize': '16px',
                                'padding': '8px',
                                'border': '1px solid #ccc',
                                'borderRadius': '4px',
                                'backgroundColor': 'white'
                            },
                            labelStyle={'display': 'block', 'marginBottom': 2}
                        ),
                        html.Button(
                            'Clear All',
                            id='clear-y-vars-button',
                            n_clicks=0,
                            style={
                                'marginTop': '5px',
                                'padding': '4px 12px',
                                'fontSize': '12px',
                                'backgroundColor': '#f8f9fa',
                                'border': '1px solid #ccc',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'width': '30%'
                            }
                        )
                    ]),
                ], width=4),
                dbc.Col([], width=2),  # Spacer to push transforms and sliders to the right
                dbc.Col([
                    html.Label("Transforms:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    dcc.Checklist(
                        id='log-checkboxes',
                        options=[
                            {'label': ' Log(X)', 'value': 'log_x'},
                            {'label': ' Log(Y)', 'value': 'log_y'}
                        ],
                        value=[],
                        labelStyle={'display': 'block', 'marginBottom': 3},
                        style={'marginBottom': '5px'}
                    ),
                    dcc.Checklist(
                        id='norm-checkboxes',
                        options=[
                            {'label': ' Normalize X', 'value': 'norm_x'},
                            {'label': ' Normalize Y', 'value': 'norm_y'}
                        ],
                        value=[],
                        labelStyle={'display': 'block', 'marginBottom': 3}
                    ),
                ], width=1),
                dbc.Col([
                    html.Label("Fit Degree:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    dcc.Slider(
                        id='degree-slider',
                        min=0, max=4, step=1, 
                        value=0 if scatter_default_fit_degree is None else scatter_default_fit_degree,
                        marks={0: 'Auto', 1: '1', 2: '2', 3: '3', 4: '4'}
                    ),
                    html.Label("Histogram Bins:", style={'fontWeight': 'bold', 'marginBottom': 5, 'marginTop': 20}),
                    dcc.Slider(
                        id='nbins-slider',
                        min=5, max=20, step=1, value=histogram_default_nbins,
                        marks={5: '5', 10: '10', 15: '15', 20: '20'}
                    ),
                ], width=2),
            ], style={'marginBottom': 15, 'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '4px'}),
        ]),
        
        # Map controls - now hidden/shown but structure preserved for callback
        html.Div(id='map-controls', style={'display': 'none'}),
        
        # Scatter view
        html.Div(id='scatter-view', children=[
            dbc.Row(id='scatter-histogram-row', children=[
                dbc.Col([
                    html.Div(id='r2-display', style={
                        'position': 'absolute',
                        'top': '50px',
                        'right': '50px',
                        'backgroundColor': 'rgba(255, 255, 255, 0.9)',
                        'padding': '8px 12px',
                        'borderRadius': '4px',
                        'border': '1px solid #ccc',
                        'zIndex': 1000,
                        'fontSize': '13px',
                        'fontFamily': font_family
                    }),
                    dcc.Graph(
                        id='scatter-plot', 
                        config={'displayModeBar': True, 'displaylogo': False, 'responsive': True},
                        style={'width': '100%', 'height': 'calc(100vh - 340px)'}  # Increased to 340px
                    )
                ], width=9, style={'height': '100%', 'position': 'relative'}, id='scatter-col'),
                dbc.Col(id='histogram-container', children=[], width=3, style={'height': '100%'}),
            ], style={'height': '100%'}),
        ], style={'height': '100%'}),
        
        # Map view - sidebar + map layout
        html.Div(id='map-view', children=[
            dbc.Row([
                # Left sidebar - 25% width with all controls stacked
                dbc.Col([
                    html.Div([
                        # Map Variables - optimized height to fit all controls
                        html.Label("Map Variables:", style={'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '15px'}),
                        dcc.Checklist(
                            id='map-var-checklist',
                            options=[{'label': col, 'value': col} for col in numeric_cols],
                            value=[map_default_variable] if map_default_variable and map_default_variable in numeric_cols else ([numeric_cols[0]] if numeric_cols else []),
                            style={
                                'height': 'calc((100vh - 120px) * 0.62)', 
                                'overflowY': 'scroll',
                                'fontSize': '14px',
                                'padding': '6px',
                                'border': '1px solid #ccc',
                                'borderRadius': '4px',
                                'backgroundColor': 'white',
                                'marginBottom': '6px'
                            },
                            labelStyle={'display': 'block', 'marginBottom': 2}
                        ),
                        html.Button(
                            'Clear All',
                            id='clear-map-vars-button',
                            n_clicks=0,
                            style={
                                'padding': '4px 10px',
                                'fontSize': '12px',
                                'backgroundColor': '#f8f9fa',
                                'border': '1px solid #ccc',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'marginBottom': '10px'
                            }
                        ),
                        
                        # Transforms
                        html.Label("Transforms:", style={'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '14px'}),
                        dcc.Checklist(
                            id='map-log-checkbox',
                            options=[{'label': ' Log', 'value': 'log'}],
                            value=[],
                            labelStyle={'display': 'block', 'marginBottom': 2},
                            style={'fontSize': '13px', 'marginBottom': '2px'}
                        ),
                        dcc.Checklist(
                            id='map-norm-checkbox',
                            options=[{'label': ' Norm', 'value': 'norm'}],
                            value=[],
                            labelStyle={'display': 'block'},
                            style={'fontSize': '13px', 'marginBottom': '10px'}
                        ),
                        
                        # Colormap
                        html.Label("Colormap:", style={'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '14px'}),
                        dcc.Dropdown(
                            id='colormap-dropdown',
                            options=[
                                {'label': 'Yellow-Green-Blue', 'value': 'YlGnBu'},
                                {'label': 'Oranges', 'value': 'Oranges'},
                                {'label': 'Oranges (inv)', 'value': 'Oranges_r'},
                                {'label': 'Red-Blue', 'value': 'RdBu'},
                                {'label': 'Blue-Red', 'value': 'RdBu_r'}
                            ],
                            value=map_default_colormap,
                            clearable=False,
                            style={'marginBottom': '10px', 'fontSize': '13px'}
                        ),
                        
                        # vmin and vmax side-by-side
                        dbc.Row([
                            dbc.Col([
                                html.Label("vmin:", style={'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '13px'}),
                                dcc.Input(
                                    id='vmin-input',
                                    type='number',
                                    placeholder='auto' if map_default_vmin is None else str(map_default_vmin),
                                    value=map_default_vmin,
                                    style={'width': '100%', 'fontSize': '13px'}
                                ),
                            ], width=6, style={'paddingRight': '5px'}),
                            dbc.Col([
                                html.Label("vmax:", style={'fontWeight': 'bold', 'marginBottom': 3, 'fontSize': '13px'}),
                                dcc.Input(
                                    id='vmax-input',
                                    type='number',
                                    placeholder='auto' if map_default_vmax is None else str(map_default_vmax),
                                    value=map_default_vmax,
                                    style={'width': '100%', 'fontSize': '13px'}
                                ),
                            ], width=6, style={'paddingLeft': '5px'}),
                        ], className='g-0'),
                    ], style={'padding': '8px', 'height': '100%', 'overflowY': 'auto'})
                ], width=3, style={'height': 'calc(100vh - 120px)', 'borderRight': '2px solid #ddd', 'paddingRight': '0', 'backgroundColor': '#f5f5f5'}),
                
                # Right side with map - 75% width
                dbc.Col([
                    dcc.Graph(
                        id='choropleth-map', 
                        config={'displayModeBar': True, 'displaylogo': False, 'responsive': True},
                        style={'width': '100%', 'height': 'calc(100vh - 120px)'}
                    )
                ], width=9, style={'height': '100%', 'paddingLeft': '0'}),
            ], style={'height': '100%'}, className='g-0'),
        ], style={'display': 'none', 'height': '100%'})
    ], fluid=True, style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})
    
    # Callbacks
    @app.callback(
        Output('map-var-checklist', 'value'),
        [Input('clear-map-vars-button', 'n_clicks')],
        [State('map-var-checklist', 'value')]
    )
    def clear_map_variables(n_clicks, current_values):
        if n_clicks > 0:
            return []
        return current_values
    
    @app.callback(
        Output('y-axis-checklist', 'value'),
        [Input('clear-y-vars-button', 'n_clicks')],
        [State('y-axis-checklist', 'value')]
    )
    def clear_y_variables(n_clicks, current_values):
        if n_clicks > 0:
            return []
        return current_values
    
    @app.callback(
        [Output('title-div', 'style'),
         Output('scatter-controls', 'style'),
         Output('map-controls', 'style'),
         Output('scatter-view', 'style'),
         Output('map-view', 'style')],
        [Input('view-tabs', 'value')]
    )
    def switch_view(tab):
        if tab == 'scatter':
            return (
                {'marginTop': '5px', 'marginBottom': '5px', 'overflow': 'hidden', 'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '4px'},
                {'display': 'block'}, 
                {'display': 'none'}, 
                {'display': 'block'}, 
                {'display': 'none'}
            )
        else:  # map
            return (
                {'marginTop': '5px', 'marginBottom': '5px', 'overflow': 'hidden', 'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '4px'},
                {'display': 'none'}, 
                {'display': 'block'}, 
                {'display': 'none'}, 
                {'display': 'block'}
            )

    @app.callback(
        [Output('scatter-plot', 'figure'),
         Output('r2-display', 'children')],
        [Input('x-axis-dropdown', 'value'),
         Input('y-axis-checklist', 'value'),
         Input('log-checkboxes', 'value'),
         Input('norm-checkboxes', 'value'),
         Input('degree-slider', 'value')]
    )
    def update_scatter(x_col, y_cols, log_checks, norm_checks, degree):
        if x_col is None or not y_cols or len(y_cols) == 0:
            return go.Figure(), ""
        
        log_x = 'log_x' in log_checks
        log_y = 'log_y' in log_checks
        norm_x = 'norm_x' in norm_checks
        norm_y = 'norm_y' in norm_checks
        fit_degree = None if degree == 0 else degree
        
        hover_cols = ['COUNTRY'] if 'COUNTRY' in df.columns else (['ADM0'] if 'ADM0' in df.columns else None)
        
        fig, r2_values = create_scatter_plot(df, x_col, y_cols, hover_cols=hover_cols, 
                                  log_x=log_x, log_y=log_y, 
                                  norm_x=norm_x, norm_y=norm_y,
                                  fit_degree=fit_degree)
        
        # Create R² display text with colored labels
        r2_elements = []
        for r2_info in r2_values:
            r2_elements.append(
                html.Span([
                    # html.Span(f"{r2_info['y_var']}: ", style={'color': r2_info['color'], 'fontWeight': 'bold'}),
                    # html.Span(f"R² = {r2_info['r2']:.3f}", style={'color': '#333'})
                    html.Span(f"R² = {r2_info['r2']:.3f}", style={'color': r2_info['color'], 'fontWeight': 'bold'})
                ])
            )
            r2_elements.append(html.Br())
        
        # Remove last <br>
        if r2_elements:
            r2_elements = r2_elements[:-1]
        
        return fig, r2_elements

    @app.callback(
        Output('histogram-container', 'children'),
        [Input('x-axis-dropdown', 'value'),
         Input('y-axis-checklist', 'value'),
         Input('log-checkboxes', 'value'),
         Input('norm-checkboxes', 'value'),
         Input('nbins-slider', 'value')]
    )
    def update_histograms(x_col, y_cols, log_checks, norm_checks, nbins):
        if x_col is None or not y_cols or len(y_cols) == 0:
            return []
        
        log_x = 'log_x' in log_checks
        log_y = 'log_y' in log_checks
        norm_x = 'norm_x' in norm_checks
        norm_y = 'norm_y' in norm_checks
        
        # Create list: Y variables reversed (so Y1 is at top), then X at bottom
        # Order from top to bottom: YN, ..., Y2, Y1, X
        all_vars = list(reversed(list(y_cols))) + [x_col]
        n_hists = len(all_vars)
        
        # Calculate height for each histogram
        height_percent = 100 / n_hists
        height_style = f'calc((100vh - 340px) * {height_percent/100} - 5px)'  # Changed to 340px
        
        histograms = []
        for idx, var in enumerate(all_vars):
            # Determine if this is X or Y variable
            is_x = (var == x_col)
            apply_log = log_x if is_x else log_y
            apply_norm = norm_x if is_x else norm_y
            
            # Get color - X uses light gray, Y variables use multi_y_colors
            if is_x:
                color = '#cccccc'  # Light gray for X variable
            else:
                y_idx = list(y_cols).index(var)
                color = multi_y_colors[y_idx % len(multi_y_colors)]
            
            fig = create_histogram(df, var, color=color,
                                 apply_log=apply_log, apply_norm=apply_norm, nbins=nbins)
            
            margin_bottom = '10px' if idx < n_hists - 1 else '0px'
            
            histograms.append(
                dcc.Graph(
                    id=f'histogram-{idx}',
                    figure=fig,
                    config={'displayModeBar': True, 'displaylogo': False, 'responsive': True},
                    style={'width': '100%', 'height': height_style, 'marginBottom': margin_bottom}
                )
            )
        
        return histograms

    @app.callback(
        [Output('choropleth-map', 'figure'),
         Output('vmin-input', 'placeholder'),
         Output('vmax-input', 'placeholder')],
        [Input('map-var-checklist', 'value'),
         Input('map-log-checkbox', 'value'),
         Input('map-norm-checkbox', 'value'),
         Input('colormap-dropdown', 'value'),
         Input('vmin-input', 'value'),
         Input('vmax-input', 'value')]
    )
    def update_map(var_cols, log_check, norm_check, colormap, vmin, vmax):
        if not var_cols or len(var_cols) == 0:
            return go.Figure(), '', ''
        
        apply_log = 'log' in log_check
        apply_norm = 'norm' in norm_check
        
        if len(var_cols) == 1:
            var_col = var_cols[0]
        else:
            selected_data = df[var_cols].copy()
            df['_temp_avg'] = selected_data.mean(axis=1, skipna=False)  # NaNs propagate
            var_col = '_temp_avg'
        
        fig, actual_vmin, actual_vmax = create_choropleth_map(
            df, var_col, 
            apply_log=apply_log, 
            apply_norm=apply_norm,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        
        if var_col == '_temp_avg' and '_temp_avg' in df.columns:
            df.drop('_temp_avg', axis=1, inplace=True)
        
        return fig, f'{actual_vmin:.3g}', f'{actual_vmax:.3g}'
    
    def open_browser():
        """Open browser after a short delay"""
        webbrowser.open_new(f'http://127.0.0.1:{port}')
    
    print("\n" + "="*60)
    print("Interactive Data Visualization Dashboard")
    print("="*60)
    print(f"\nBrowser will open automatically to: http://127.0.0.1:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # if openBrowser:
    Timer(1.5, open_browser).start()
    # else:
    app.run(debug=debug, port=port, use_reloader=False)