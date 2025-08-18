import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import os # type: ignore
import cartopy.util as cutil # type: ignore
import cartopy.crs as ccrs # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as mpatches # type: ignore
import cftime # type: ignore
import cartopy.feature as cfeature # type: ignore
from matplotlib.patches import Rectangle # type: ignore
import regionmask # type: ignore
from fwdp import computeFireWeatherIndices

# =====================================================================================
# SETTINGS

# Global
lat_bot_glb = -90
lat_top_glb = 90
lon_west_glb = 0
lon_east_glb = 360

# Western North America (from Touma et al., 2023 - Table S1: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003626)
lat_bot_wna = 30
lat_top_wna = 45
lon_west_wna = 360-125
lon_east_wna = 360-115.5

# Northeast Brazil (from Touma et al., 2023 - Table S1: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003626)
lat_bot_nbr = -17.5
lat_top_nbr = 2.5
lon_west_nbr = 360-60
lon_east_nbr = 360-33

# Western Amazon (from Touma et al., 2023 - Table S1: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003626)
lat_bot_wamz = -14.5
lat_top_wamz = -0.5
lon_west_wamz = 360-82
lon_east_wamz = 360-60

# Mediterranean (from Touma et al., 2023 - Table S1: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003626)
lat_bot_med = 36
lat_top_med = 47
lon_west_med = 360-10.5
lon_east_med = 48

# West Central Africa (from Touma et al., 2023 - Table S1: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003626)
lat_bot_wca = -8
lat_top_wca = 1
lon_west_wca = 8
lon_east_wca = 28

# Northern Australia (from Touma et al., 2023 - Table S1: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EF003626)
lat_bot_nau = -20
lat_top_nau = -5
lon_west_nau = 120
lon_east_nau = 150

# East Asia (from Wilcox et al., 2023 - Figure 2: https://gmd.copernicus.org/articles/16/4451/2023/)
lat_bot_eas = 20
lat_top_eas = 53
lon_west_eas = 95
lon_east_eas = 133

# South Asia (from Wilcox et al., 2023 - Figure 2: https://gmd.copernicus.org/articles/16/4451/2023/)
lat_bot_sas = 5
lat_top_sas = 35
lon_west_sas = 65
lon_east_sas = 95

# Africa & Mid-East (from Wilcox et al., 2023 - Figure 2: https://gmd.copernicus.org/articles/16/4451/2023/)
lat_bot_afr = -35
lat_top_afr = 35
lon_west_afr = 360-20
lon_east_afr = 60

# NAM (from Wilcox et al., 2023 - Figure 2: https://gmd.copernicus.org/articles/16/4451/2023/)
lat_bot_nam = 25
lat_top_nam = 70
lon_west_nam = 360-150
lon_east_nam = 360-45

# EU (from Wilcox et al., 2023 - Figure 2: https://gmd.copernicus.org/articles/16/4451/2023/)
lat_bot_eur = 35
lat_top_eur = 70
lon_west_eur = 360-20
lon_east_eur = 45

# # Western (Eurasian) Steppe (estimated from https://www.britannica.com/place/the-Steppe)
# lat_bot_wst = 40
# lat_top_wst = 55
# lon_west_wst = 10
# lon_east_wst = 80

# Western (Eurasian) Steppe (estimated from https://www.britannica.com/place/the-Steppe, refined to where we see strong FWI signals outside of the Mediterranean)
lat_bot_wst = 47
lat_top_wst = 55
lon_west_wst = 30
lon_east_wst = 80

# Inner Mongolia Steppe (estimated from https://www.britannica.com/place/the-Steppe, refined to where we see strong FWI signals outside of the Mediterranean)
lat_bot_imst = 33
lat_top_imst = 50
lon_west_imst = 97
lon_east_imst = 120

# Central Asia
lat_bot_cna = 35
lat_top_cna = 47
lon_west_cna = 50
lon_east_cna = 80

# Southwestern NA
lat_bot_swna = 23
lat_top_swna = 40
lon_west_swna = 360-115
lon_east_swna = 360-93

# Southern Africa
lat_bot_saf = -35
lat_top_saf = -10
lon_west_saf = 10
lon_east_saf = 40

# Australian Interior
lat_bot_ausi = -30
lat_top_ausi = -20
lon_west_ausi = 120
lon_east_ausi = 145

# Southeastern Australia
lat_bot_seaus = -43
lat_top_seaus = -30
lon_west_seaus = 130
lon_east_seaus = 153

# West Coast of Australia
lat_bot_wcaus = -35
lat_top_wcaus = -20
lon_west_wcaus = 110
lon_east_wcaus = 120

# East Coast of Australia
lat_bot_ecaus = -30
lat_top_ecaus = -20
lon_west_ecaus = 145
lon_east_ecaus = 153

# Eastern NA
lat_bot_ena = 21.46
lat_top_ena = 56.03
lon_west_ena = 267.5
lon_east_ena = 292.1

# Eastern Europe
lat_bot_eeu = 47
lat_top_eeu = 66.4
lon_west_eeu = 24
lon_east_eeu = 76

# Australia
lat_bot_aus = -44.3
lat_top_aus = -10.95
lon_west_aus = 113.07
lon_east_aus = 154.5

# Amazon / South America
lat_bot_amz = -27.64
lat_top_amz = -0.48
lon_west_amz = 276
lon_east_amz = 325.73

# Southeast Asia
lat_bot_sea = -11.89
lat_top_sea = 29.11
lon_west_sea = 91.79
lon_east_sea = 152.64


# =====================================================================================
# FUNCTIONS

# (1) Read zarr data
def read_zarr(model, experiment, variable,
              start_analysis=cftime.DatetimeNoLeap(2021, 1, 1, 12, 0, 0, 0, has_year_zero=True),
              end_analysis=cftime.DatetimeNoLeap(2050, 12, 31, 0, 0, 0, 0, has_year_zero=True),
              lat_bot=-90, lat_top=90, lon_west=0, lon_east=360,
              base_dir="/projects/dgs/persad_research/SIMULATION_DATA/ZARR/"):
    """
    Unified function to read zarr data for any model, experiment, and variable.
    
    Parameters:
    -----------

    model : str
        Model name (e.g., 'CESM2', 'NorESM2-LM')
    experiment : str
        Experiment name (e.g., 'historical', 'ssp370-126aer')
    variable : str
        Variable name (e.g., 'FWI', 'hurs', 'pr')
    start_analysis, end_analysis : cftime.DatetimeNoLeap
        Start and end dates for analysis
    lat_bot, lat_top, lon_west, lon_east : float
        Geographical boundaries
    base_dir : str
        Base directory for data files
        
    Returns:
    --------
    xarray.DataArray
        The requested data subset
    """
    # Handle historical CESM2 variable name mapping
    var_mapping = {
        'hurs': 'RHREFHT',
        'pr': 'PRECT',
        'sfcWind': 'U10',
        'tasmax': 'tasmax'
    }
    
    # Determine directory and variable name
    if variable == "FWI":
        dir_path = f"{base_dir}RAMIP/FWI_OUTPUTS/{model}/FWI_value/"
        actual_var = "FWI"
    else:
        dir_path = f"{base_dir}RAMIP/SIM_VARIABLES/"
        if model == "CESM2" and experiment == "historical":
            actual_var = var_mapping.get(variable, variable)
        else:
            actual_var = variable
    
    # Construct file path
    file_path = f"{dir_path}{model}_{experiment}_day_{variable}.zarr"
    
    # Read and process data
    da = xr.open_zarr(file_path)[actual_var]
    da = da.sel(time=slice(start_analysis, end_analysis),
                lat=slice(lat_bot, lat_top),
                lon=slice(lon_west, lon_east))
    
    return da.assign_coords({"model": model})

# (2) Output FWI data
def output_FWI_data(model, experiment, 
                    output_type='FWI',  # Options: 'FWI', 'threshold', 'exceedance'
                    start_analysis=cftime.DatetimeNoLeap(2021, 1, 1, 12, 0, 0, 0, has_year_zero=True),
                    end_analysis=cftime.DatetimeNoLeap(2050, 12, 31, 0, 0, 0, 0, has_year_zero=True),
                    corrected=False,
                    base_dir="/projects/dgs/persad_research/SIMULATION_DATA/ZARR/",
                    threshold_type=None,  # 'percentile' or 'absolute'
                    threshold_value=None):
    """
    Unified function to calculate and save different types of FWI data
    
    Parameters:
    -----------
    model : str
        Model name (e.g., 'CESM2', 'NorESM2-LM', 'MRI-ESM2-0', 'SPEAR')
    experiment : str
        Experiment name (e.g., 'historical', 'ssp370-126aer')
    output_type : str
        Type of output to generate ('FWI', 'threshold', or 'exceedance')
    start_analysis, end_analysis : cftime.DatetimeNoLeap
        Start and end dates for analysis
    corrected : bool
        Whether to use bias-corrected temperature data
    base_dir : str
        Base directory for data files
    threshold_type : str, optional
        Type of threshold ('percentile' or 'absolute')
    threshold_value : float, optional
        Value for threshold calculation
        
    Notes:
    ------
    Unit conversions applied:
    - tasmax: Kelvin to Celsius (subtract 273.15)
    - pr: kg/m²/s to mm/day (multiply by 86400)
    - sfcWind: m/s to km/hr (multiply by 3.6)
    """
    # Model-specific chunk settings
    chunk_settings = {
        "CESM2": dict(time=-1, lat=48, lon=72, member=1),
        "NorESM2-LM": dict(time=-1, lat=24, lon=36, member=1),
        "MRI-ESM2-0": dict(time=-1, lat=40, lon=80, member=1),
        "SPEAR": dict(time=-1, lat=45, lon=72, member=1)
    }

    # Validate model
    if model not in chunk_settings:
        raise ValueError(f"Unsupported model: {model}. Must be one of {list(chunk_settings.keys())}")

    # Validate output_type and threshold parameters
    if output_type not in ['FWI', 'threshold', 'exceedance']:
        raise ValueError("output_type must be 'FWI', 'threshold', or 'exceedance'")
        
    if output_type in ['threshold', 'exceedance']:
        if not threshold_type or not threshold_value:
            raise ValueError("threshold_type and threshold_value required for threshold or exceedance output")
        if threshold_type not in ['percentile', 'absolute']:
            raise ValueError("threshold_type must be 'percentile' or 'absolute'")
        if threshold_type == 'percentile' and not (0 <= threshold_value <= 100):
            raise ValueError("percentile threshold_value must be between 0 and 100")

    # Read and process input variables with unit conversions
    # ! Please refer to utilities_CESM2_NorESM2_v1_2.py for the unit conversions, especially CESM2 historical (starts with m/s for pr instead of kg/m²/s)
    tasmax_suffix = "_corrected" if corrected else ""
    tasmax = read_zarr(model, experiment, f'tasmax{tasmax_suffix}', 
                      start_analysis, end_analysis) - 273.15  # K to °C 
    hurs = read_zarr(model, experiment, 'hurs', 
                    start_analysis, end_analysis)  # Already in %
    pr = read_zarr(model, experiment, 'pr', 
                  start_analysis, end_analysis) * 86400  # kg/m²/s to mm/day
    sfcWind = read_zarr(model, experiment, 'sfcWind', 
                       start_analysis, end_analysis) * 3.6  # m/s to km/hr

    print(f"Data loaded for {model} {experiment}")

    # Calculate base FWI
    FWI = computeFireWeatherIndices(tasmax, hurs, pr, sfcWind)['FWI']
    print("FWI calculation completed")
    
    # Determine output path and prepare data based on output_type
    if output_type == 'FWI':
        output_path = f"{base_dir}RAMIP/FWI_OUTPUTS/{model}/FWI_value/"
        output_data = FWI
        
    elif output_type == 'threshold':
        output_path = f"{base_dir}RAMIP/FWI_OUTPUTS/{model}/FWI_threshold_{threshold_type}/"
        
        if threshold_type == 'percentile':
            output_data = FWI.quantile(threshold_value/100, dim='time')
            print(f"Calculated {threshold_value}th percentile threshold")
        else:  # absolute threshold
            output_data = xr.full_like(FWI.isel(time=0), threshold_value)
            print(f"Set absolute threshold of {threshold_value}")
            
    else:  # exceedance
        output_path = f"{base_dir}RAMIP/FWI_OUTPUTS/{model}/FWI_threshold_{threshold_type}_exceedance/"
        
        # Get threshold
        if threshold_type == 'percentile':
            threshold = FWI.quantile(threshold_value/100, dim='time')
            print(f"Calculated {threshold_value}th percentile threshold")
        else:  # absolute threshold
            threshold = xr.full_like(FWI.isel(time=0), threshold_value)
            print(f"Using absolute threshold of {threshold_value}")
            
        # Calculate exceedances
        output_data = (FWI > threshold).astype(int)
        print("Calculated exceedances")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare filename
    if corrected:
        filename = f"{model}_{experiment}_corrected_day_FWI"
    else:
        filename = f"{model}_{experiment}_day_FWI"
        
    if output_type != 'FWI':
        filename += f"_{threshold_type}_{threshold_value}"
    filename += ".zarr"
    
    # Save to zarr with appropriate chunking
    output_data.to_zarr(
        f"{output_path}{filename}",
        mode="w",
        consolidated=True,
        encoding={var: {"chunks": chunk_settings[model]} for var in output_data.data_vars}
    )
    
    print(f"Data saved to {output_path}{filename}")
    
    # Clean up to free memory
    del tasmax, hurs, pr, sfcWind, FWI, output_data

# (3) Weighted horizontal average
def weighted_horizontal_avg(da, ensemble = True, time = True):
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    if ensemble == True and time == True:
        if "member" in da.dims:
            weighted_mean = da_weighted.mean(("lon", "lat", "member", "time"))
        elif "model" in da.dims:
            weighted_mean = da_weighted.mean(("lon", "lat", "model", "time"))
    elif ensemble == True and time == False:
        if "member" in da.dims:
            weighted_mean = da_weighted.mean(("lon", "lat", "member"))
        elif "model" in da.dims:
            weighted_mean = da_weighted.mean(("lon", "lat", "model"))
    elif ensemble == False and time == True:
        weighted_mean = da_weighted.mean(("lon", "lat", "time"))
    elif ensemble == False and time == False:
        weighted_mean = da_weighted.mean(("lon", "lat"))
    return weighted_mean

# (4) Seasonal mean
def season_mean(ds, calendar="noleap"):
    """
    Calculate weighted seasonal means taking into account the number of days in each month.
    
    Parameters:
    ds (xarray.DataArray): Input data with a time dimension
    calendar (str): Calendar type, default is "noleap"
    
    Returns:
    xarray.DataArray: Seasonal means weighted by the number of days in each month
    """
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average with skipna=True
    seasonal = (ds * weights).groupby("time.season").sum(dim="time", skipna=True)
    
    # Reorder seasons
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    return seasonal.reindex(season=season_order)

# (5) Calculate temporal averages
def calculate_temporal_averages(data):
    """
    Calculate daily, annual, seasonal, and monthly averages for a dataset
    
    Parameters:
    data (xarray.DataArray): Input temperature data with a time dimension
    
    Returns:
    dict: Dictionary containing daily, annual, seasonal, and monthly averages
    """
    
    # Annual average with skipna=True
    annual_mean = data.groupby('time.year').mean('time', skipna=True)
    annual_mean = weighted_horizontal_avg(annual_mean, member=False, time=False)
    
    # Seasonal averages using the weighted method
    seasonal_mean = season_mean(data)
    seasonal_mean = weighted_horizontal_avg(seasonal_mean, member=False, time=False)
    
    # Monthly averages with skipna=True
    monthly_mean = data.groupby('time.month').mean('time', skipna=True)
    monthly_mean = weighted_horizontal_avg(monthly_mean, member=False, time=False)

    # Daily average using resample
    daily_mean = data
    daily_mean = weighted_horizontal_avg(daily_mean, member=False, time=False)
    
    return {
        'annual': annual_mean,
        'seasonal': seasonal_mean,
        'monthly': monthly_mean,
        'daily': daily_mean
    }

# (6) Calculate exceedance days
def calculate_exceedance_days(threshold,
                            data,
                            frequency='annual',
                            horizontal_avg=False,
                            ensemble=True):
    """
    Calculate temporal mean of exceedance days for each grid point.
    
    Parameters:
    -----------
    threshold : float or xarray.DataArray
        Threshold value(s) for defining exceedance
    data : xarray.DataArray
        Input variable data
    frequency : str, optional
        Frequency of aggregation. Options:
        'annual' - yearly
        'seasonal' - DJF, MAM, JJA, SON
        'monthly' - each month
    ensemble : bool, optional
        Whether to average across ensemble members/models
        
    Returns:
    --------
    xarray.DataArray
        Temporal mean of exceedance days at specified frequency
    """
    
    # Get exceedance days
    exceedance_boolean = data > threshold

    # Calculate exceedance counts based on specified frequency
    if frequency == 'annual':
        exceedance_count = exceedance_boolean.resample(time='Y').sum(dim=['time'], skipna=True)
        
    elif frequency == 'seasonal':
        # Sum the exceedance days by season instead of averaging
        exceedance_count = exceedance_boolean.groupby('time.season').sum(dim='time', skipna=True)
        # Reorder seasons
        season_order = ['DJF', 'MAM', 'JJA', 'SON']
        exceedance_count = exceedance_count.reindex(season=season_order)
        
    elif frequency == 'monthly':
        # Group by month and calculate mean across years
        exceedance_count = exceedance_boolean.groupby('time.month').sum('time', skipna=True)
        
    else:
        raise ValueError("frequency must be 'annual', 'seasonal', or 'monthly'")

    # Initialize result with the original count data
    if horizontal_avg == True:
        result = weighted_horizontal_avg(exceedance_count, ensemble=ensemble, time=False)
    else:
        result = exceedance_count

    # Handle ensemble averaging if requested
    if ensemble:
        if "member" in result.dims:
            result = result.mean('member', skipna=True)
        elif "model" in result.dims:
            result = result.mean('model', skipna=True)

    return result

# (7) Create global map
def create_global_map(data, 
                     projection=ccrs.Robinson(),
                     title="Global Map",
                     colormap='RdBu_r',
                     colorbar_title="Change",
                     textbox_text=None,
                     figsize=(10.5, 6),
                     vmin=None,
                     vmax=None,
                     extend='both',
                     colorbar_levels=None,
                     contour_levels=None,
                     hatching='///',
                     regional_boundaries=True,
                     hatching_style='overlay',
                     hatching_data=None,
                     colorbar_extend=None,
                     show_gridlines=True,
                     ramip_regions=False):
    """
    Create a global map visualization similar to climate change impact maps.
    
    Parameters:
    -----------
    data : xarray.DataArray or list of xarray.DataArrays
        If DataArray: 2D DataArray with lat/lon coordinates for filled contour plot
        If list: [fill_data, contour_data] where fill_data is for colors and 
                 contour_data is for contour lines (contour_data is optional)
    projection : cartopy projection, optional
        Map projection to use (default: Robinson)
    title : str, optional
        Title for the plot (default: "Global Map")
    colormap : str, optional
        Matplotlib colormap name (default: 'RdBu_r')
    colorbar_title : str, optional
        Title for the colorbar (default: "Change")
    textbox_text : str, optional
        Text to display in the bottom-left textbox (supports f-string format)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10.5, 6))
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    extend : str, optional
        Colorbar extension ('both', 'min', 'max', 'neither') - DEPRECATED, use colorbar_extend
    colorbar_levels : array-like, optional
        Specific levels for the colorbar/filled contours
    contour_levels : array-like, optional
        Specific levels for contour lines
    hatching : str, optional
        Hatching pattern for masked areas (default: '///', set to None to disable)
    regional_boundaries : str or bool, optional
        Type of boundaries to show. Options: True or 'countries' (country boundaries), 
        'ar6' (IPCC AR6 WG1 reference regions), False or 'none' (no boundaries) 
        (default: True)
    hatching_style : str, optional
        How to display hatched areas: 'overlay' (hatching over colors) or 
        'white' (white background with hatching) (default: 'overlay')
    hatching_data : xarray.DataArray, optional
        Data array indicating where to apply hatching. Hatching will be applied 
        where hatching_data is NOT NaN (i.e., only over valid hatching indicator areas).
    colorbar_extend : str, optional
        Controls colorbar arrows: 'min' (arrow at bottom), 'max' 
        (arrow at top), 'both' (arrows at both ends), 'neither' (no arrows).
        If None, uses the 'extend' parameter for backward compatibility.
    show_gridlines : bool, optional
        Whether to display map gridlines (default: True)
    ramip_regions : bool, str, or list, optional
        Whether to display RAMIP emission region boxes. Options:
        - False: No RAMIP regions (default)
        - True: All 4 groups of RAMIP regions
        - 'east_asia': East Asia region only
        - 'south_asia': South Asia region only  
        - 'africa_mideast': Africa & Middle East region only
        - 'north_america_europe': North America & Europe regions only
        - list: Combination of region names, e.g. ['east_asia', 'south_asia']
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    
    Example:
    --------
    # Single dataset example with RAMIP regions
    lons = np.linspace(-180, 180, 144)
    lats = np.linspace(-90, 90, 72)
    fill_data = xr.DataArray(
        np.random.randn(72, 144) * 20,
        coords={'lat': lats, 'lon': lons},
        dims=['lat', 'lon']
    )
    
    fig, ax = create_global_map(
        data=fill_data,
        title="Effect of Aerosol Emission Reduction",
        colorbar_title="Δ Annual High Fire Danger Days",
        textbox_text="2.29 days",
        ramip_regions=True,  # Show all RAMIP regions, or specify: 'east_asia', 'south_asia', etc.
        regional_boundaries='ar6',
        show_gridlines=False
    )
    plt.show()
    """

    # Handle input data - can be single DataArray or list
    if isinstance(data, list):
        fill_data = data[0]
        contour_data = data[1] if len(data) > 1 else None
    else:
        fill_data = data
        contour_data = None
    
    # Create figure and axis with specified projection
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = plt.axes(projection=projection)
    
    # Set global extent
    ax.set_global()
    
    # Add map features - coastlines always shown
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    
    # Add boundaries based on user selection
    if regional_boundaries is True or regional_boundaries == 'countries':
        # Show country boundaries (default behavior)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', alpha=0.7)
    elif regional_boundaries == 'ar6':
        # Show IPCC AR6 WG1 reference regions
        ar6_regions = regionmask.defined_regions.ar6.land
        # Use cartopy-compatible plotting
        ar6_regions.plot_regions(ax=ax, 
                                line_kws={'linewidth': 0.8, 'color': 'black', 'alpha': 0.4},
                                add_label=False)
    # If False, 'none', or any other value, no additional boundaries are added
    
    # Handle fill data coordinates - ensure they're named correctly
    if 'latitude' in fill_data.coords:
        fill_data = fill_data.rename({'latitude': 'lat'})
    if 'longitude' in fill_data.coords:
        fill_data = fill_data.rename({'longitude': 'lon'})
    
    # Determine colorbar extension
    if colorbar_extend is not None:
        # Use the new parameter (now matches matplotlib's extend options directly)
        if colorbar_extend in ['min', 'max', 'both', 'neither']:
            extend_param = colorbar_extend
        else:
            raise ValueError("colorbar_extend must be 'min', 'max', 'both', or 'neither'")
    else:
        # Fall back to the old extend parameter for backward compatibility
        extend_param = extend
    
    # Set color limits if not provided
    if vmin is None or vmax is None:
        data_range = np.nanmax(np.abs(fill_data.values))
        if vmin is None:
            vmin = -data_range
        if vmax is None:
            vmax = data_range
    
    # Create the main filled contour plot using contourf
    if colorbar_levels is not None:
        # Use provided levels
        im = ax.contourf(fill_data.lon, fill_data.lat, fill_data.values, 
                        levels=colorbar_levels,
                        transform=ccrs.PlateCarree(),
                        cmap=colormap,
                        vmin=vmin, 
                        vmax=vmax,
                        extend=extend_param)
    else:
        # Generate automatic levels for contourf
        im = ax.contourf(fill_data.lon, fill_data.lat, fill_data.values,
                        transform=ccrs.PlateCarree(),
                        cmap=colormap,
                        vmin=vmin, 
                        vmax=vmax,
                        extend=extend_param)
    
    # Add contour lines if contour data is provided
    if contour_data is not None:
        # Handle contour data coordinates
        if 'latitude' in contour_data.coords:
            contour_data = contour_data.rename({'latitude': 'lat'})
        if 'longitude' in contour_data.coords:
            contour_data = contour_data.rename({'longitude': 'lon'})
        
        # Create contour lines
        if contour_levels is not None:
            cs = ax.contour(contour_data.lon, contour_data.lat, contour_data.values,
                           levels=contour_levels,
                           transform=ccrs.PlateCarree(),
                           colors='black',
                           linewidths=0.8,
                           alpha=0.7)
        else:
            cs = ax.contour(contour_data.lon, contour_data.lat, contour_data.values,
                           transform=ccrs.PlateCarree(),
                           colors='black',
                           linewidths=0.8,
                           alpha=0.7)
        
        # Optionally add contour labels
        ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
    
    # Add hatching only where hatching_data specifies
    if hatching is not None and hatching_data is not None:
        # Handle coordinates for hatching data
        if 'latitude' in hatching_data.coords:
            hatching_data = hatching_data.rename({'latitude': 'lat'})
        if 'longitude' in hatching_data.coords:
            hatching_data = hatching_data.rename({'longitude': 'lon'})
        
        # Create hatching mask that uses hatching_data 
        hatching_data_valid = ~np.isnan(hatching_data.values)  # True where hatching_data is not NaN
        hatch_mask = hatching_data_valid 
        
        if np.any(hatch_mask):
            # Create meshgrid for hatching
            X, Y = np.meshgrid(hatching_data.lon, hatching_data.lat)
            
            if hatching_style == 'overlay':
                # Overlay hatching (shows colors underneath)
                ax.contourf(X, Y, hatch_mask.astype(int), 
                           levels=[0.5, 1.5], 
                           colors='none', 
                           hatches=[hatching], 
                           transform=ccrs.PlateCarree())
            elif hatching_style == 'white':
                # White areas with hatching (covers the filled contours)
                ax.contourf(X, Y, hatch_mask.astype(int),
                           levels=[0.5, 1.5], 
                           colors=['white'],
                           hatches=[hatching],
                           transform=ccrs.PlateCarree(),
                           alpha=1.0,
                           zorder=5)
    
    # Add RAMIP emission region boxes
    if ramip_regions:
        # Determine which regions to show
        regions_to_show = []
        
        if ramip_regions is True:
            # Show all regions
            regions_to_show = ['east_asia', 'south_asia', 'africa_mideast', 'north_america_europe']
        elif isinstance(ramip_regions, str):
            # Single region specified
            regions_to_show = [ramip_regions]
        elif isinstance(ramip_regions, (list, tuple)):
            # Multiple regions specified
            regions_to_show = list(ramip_regions)
        
        # Draw the specified regions
        for region in regions_to_show:
            if region == 'east_asia':
                # East Asia
                ax.add_patch(mpatches.Rectangle(
                    xy=[lon_west_eas, lat_bot_eas], 
                    width=(lon_east_eas - lon_west_eas), 
                    height=(lat_top_eas - lat_bot_eas),
                    facecolor='none',
                    linestyle='--',
                    linewidth=2.0,
                    edgecolor='black',
                    alpha=1,
                    transform=ccrs.PlateCarree())
                )
            
            elif region == 'south_asia':
                # South Asia
                ax.add_patch(mpatches.Rectangle(
                    xy=[lon_west_sas, lat_bot_sas], 
                    width=(lon_east_sas - lon_west_sas), 
                    height=(lat_top_sas - lat_bot_sas),
                    facecolor='none',
                    linestyle='--',
                    linewidth=2.0,
                    edgecolor='black',
                    alpha=1,
                    transform=ccrs.PlateCarree())
                )
            
            elif region == 'africa_mideast':
                # Africa & Mid-East (crosses dateline) - use single rectangle with proper wrapping
                ax.add_patch(mpatches.Rectangle(
                    xy=[lon_west_afr, lat_bot_afr], 
                    width=(lon_east_afr + (360 - lon_west_afr)), 
                    height=(lat_top_afr - lat_bot_afr),
                    facecolor='none',
                    linestyle='--',
                    linewidth=2.0,
                    edgecolor='black',
                    alpha=1,
                    transform=ccrs.PlateCarree())
                )
            
            elif region == 'north_america_europe':
                # North America
                ax.add_patch(mpatches.Rectangle(
                    xy=[lon_west_nam, lat_bot_nam], 
                    width=(lon_east_nam - lon_west_nam), 
                    height=(lat_top_nam - lat_bot_nam),
                    facecolor='none',
                    linestyle='--',
                    linewidth=2.0,
                    edgecolor='black',
                    alpha=1,
                    transform=ccrs.PlateCarree())
                )
                
                # Europe (crosses dateline) - use single rectangle with proper wrapping
                ax.add_patch(mpatches.Rectangle(
                    xy=[lon_west_eur, lat_bot_eur], 
                    width=(lon_east_eur + (360 - lon_west_eur)), 
                    height=(lat_top_eur - lat_bot_eur),
                    facecolor='none',
                    linestyle='--',
                    linewidth=2.0,
                    edgecolor='black',
                    alpha=1,
                    transform=ccrs.PlateCarree())
                )
    
    # Add gridlines conditionally
    if show_gridlines:
        gl = ax.gridlines(draw_labels=False, linestyle='-', alpha=0.3, color='gray')
    
    # Create colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.04, shrink=0.8, aspect=20,
                       extend=extend_param)
    cbar.set_label(colorbar_title, fontsize=18, fontweight='regular')
    cbar.ax.tick_params(labelsize=14)
    
    # Add title
    plt.title(title, fontsize=22, fontweight='regular', pad=15)
    
    # Add textbox in bottom-left corner if text is provided
    if textbox_text is not None:
        # Add white background rectangle
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor='black', linewidth=1.0)
        
        ax.text(0.02, 0.02, textbox_text, transform=ax.transAxes,
                fontsize=14, fontweight='regular',
                verticalalignment='center', horizontalalignment='left',
                bbox=bbox_props, zorder=10)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

# (8) Create global map grid
def create_global_map_grid(data_list, 
                          rows=2, 
                          cols=2,
                          main_title="Global Map Grid",
                          projection=ccrs.Robinson(),
                          titles=None,
                          colormaps='RdBu_r',
                          colorbar_titles="Change",
                          textbox_texts=None,
                          figsize=None,
                          vmins=None,
                          vmaxs=None,
                          extends='both',
                          colorbar_levels=None,
                          contour_levels=None,
                          hatchings='///',
                          regional_boundaries=True,
                          hatching_styles='overlay',
                          hatching_data=None,
                          show_gridlines=True,
                          ramip_regions=False,
                          subplot_spacing={'hspace': 0.3, 'wspace': 0.1},
                          colorbar_spacing=0.05):
    """
    Create a grid of global map visualizations similar to climate change impact maps.
    
    Parameters:
    -----------
    data_list : list
        List of data for each subplot. Each item can be:
        - xarray.DataArray: 2D DataArray with lat/lon coordinates for filled contour plot
        - tuple/list: (fill_data, contour_data) where fill_data is for colors and 
                      contour_data is for contour lines
    rows : int
        Number of rows in the grid (default: 2)
    cols : int  
        Number of columns in the grid (default: 2)
    main_title : str, optional
        Main title for the entire grid (default: "Global Map Grid")
    projection : cartopy projection or list, optional
        Map projection(s) to use. Can be single projection for all maps or list of projections
    titles : str or list, optional
        Title(s) for individual subplots. Can be single string or list
    colormaps : str or list, optional
        Colormap(s) to use. Can be single colormap or list (default: 'RdBu_r')
    colorbar_titles : str or list, optional
        Title(s) for colorbars. Can be single string or list (default: "Change")
    textbox_texts : str, list, or None, optional
        Text(s) for textboxes in subplots. Can be single string, list, or None
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, auto-calculated based on grid size
    vmins : float or list, optional
        Minimum value(s) for color scale. Can be single value or list
    vmaxs : float or list, optional  
        Maximum value(s) for color scale. Can be single value or list
    extends : str or list, optional
        Colorbar extension(s): 'min', 'max', 'both', 'neither'. Can be single value or list
    colorbar_levels : array-like or list, optional
        Specific level(s) for colorbars. Can be single array or list of arrays
    contour_levels : array-like or list, optional
        Specific level(s) for contour lines. Can be single array or list of arrays
    hatchings : str or list, optional
        Hatching pattern(s). Can be single pattern or list (default: '///')
    regional_boundaries : str, bool, or list, optional
        Boundary type(s) to show. Can be single value or list
    hatching_styles : str or list, optional
        Hatching style(s). Can be single style or list (default: 'overlay')
    hatching_data : xarray.DataArray, list, or None, optional
        Hatching data for each subplot. Can be single DataArray, list, or None
    extends : str or list, optional
        Colorbar extension(s): 'min', 'max', 'both', 'neither'. Can be single value or list
    show_gridlines : bool or list, optional
        Whether to show gridlines. Can be single bool or list (default: True)
    ramip_regions : bool, str, list, or list of lists, optional
        RAMIP regions to show. Can be single value or list of values for each subplot
    subplot_spacing : dict, optional
        Spacing parameters for subplots: {'hspace': vertical, 'wspace': horizontal}
    colorbar_spacing : float, optional
        Additional space below each subplot for colorbar (default: 0.05)
        
    Returns:
    --------
    fig : matplotlib figure object
    axes : array of matplotlib axis objects
    
    Example:
    --------
    # Create sample data
    lons = np.linspace(-180, 180, 144)
    lats = np.linspace(-90, 90, 72)
    
    data1 = xr.DataArray(np.random.randn(72, 144) * 20,
                        coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
    data2 = xr.DataArray(np.random.randn(72, 144) * 15,
                        coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
    data3 = xr.DataArray(np.random.randn(72, 144) * 25,
                        coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
    data4 = xr.DataArray(np.random.randn(72, 144) * 10,
                        coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
    
    # Example with 5 maps in 2 columns (3 rows)
    fig, axes = create_global_map_grid(
        data_list=[data1, data2, data3, data4, data5],
        rows=3, cols=2,  # 6 subplot positions, but only 5 maps (last position will be empty)
        main_title="Fire Weather Changes - 5 Scenarios",
        titles=["Scenario A", "Scenario B", "Scenario C", "Scenario D", "Scenario E"]
    )
    plt.show()
    """
    
    # Validate inputs
    n_plots = len(data_list)
    max_plots = rows * cols
    if n_plots > max_plots:
        raise ValueError(f"Number of data items ({n_plots}) exceeds grid capacity ({max_plots})")
    if n_plots == 0:
        raise ValueError("data_list cannot be empty")
    
    # Helper function to convert single values to lists
    def _ensure_list(param, default_val, n_items):
        if param is None:
            return [default_val] * n_items
        elif isinstance(param, (list, tuple)):
            # It's a list/tuple - check if it's meant to be per-subplot or single value
            if len(param) == n_items:
                return list(param)
            elif len(param) == 1:
                return [param[0]] * n_items
            else:
                raise ValueError(f"Parameter list length ({len(param)}) must be either 1 or {n_items}")
        elif isinstance(param, np.ndarray):
            # For numpy arrays, treat as single parameter to be applied to all subplots
            return [param] * n_items
        else:
            # Single value - apply to all subplots
            return [param] * n_items
    
    # Convert all parameters to lists
    projections = _ensure_list(projection, ccrs.Robinson(), n_plots)
    title_list = _ensure_list(titles, "Global Map", n_plots)
    colormap_list = _ensure_list(colormaps, 'RdBu_r', n_plots)
    colorbar_title_list = _ensure_list(colorbar_titles, "Change", n_plots)
    textbox_text_list = _ensure_list(textbox_texts, None, n_plots)
    vmin_list = _ensure_list(vmins, None, n_plots)
    vmax_list = _ensure_list(vmaxs, None, n_plots)
    extend_list = _ensure_list(extends, 'both', n_plots)
    colorbar_level_list = _ensure_list(colorbar_levels, None, n_plots)
    contour_level_list = _ensure_list(contour_levels, None, n_plots)
    hatching_list = _ensure_list(hatchings, '///', n_plots)
    regional_boundary_list = _ensure_list(regional_boundaries, True, n_plots)
    hatching_style_list = _ensure_list(hatching_styles, 'overlay', n_plots)
    hatching_data_list = _ensure_list(hatching_data, None, n_plots)
    show_gridlines_list = _ensure_list(show_gridlines, True, n_plots)
    ramip_regions_list = _ensure_list(ramip_regions, False, n_plots)
    
    # Calculate figure size if not provided
    if figsize is None:
        # Base size per subplot, adjusted for colorbars and spacing
        base_width = 5.0
        base_height = 3.5
        figsize = (cols * base_width, rows * (base_height + colorbar_spacing * 8))
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Calculate subplot positions to leave room for individual colorbars
    subplot_height = (1.0 - subplot_spacing['hspace']) / rows - colorbar_spacing
    subplot_width = (1.0 - subplot_spacing['wspace']) / cols
    
    axes = []
    
    for i in range(n_plots):
        row = i // cols
        col = i % cols
        
        # Calculate subplot position
        left = col * (subplot_width + subplot_spacing['wspace'] / cols) + subplot_spacing['wspace'] / (2 * cols)
        bottom = (rows - row - 1) * (subplot_height + colorbar_spacing + subplot_spacing['hspace'] / rows) + colorbar_spacing + subplot_spacing['hspace'] / (2 * rows)
        
        # Create subplot with cartopy projection
        ax = fig.add_subplot(rows, cols, i + 1, projection=projections[i])
        axes.append(ax)
        
        # Get current data
        current_data = data_list[i]
        
        # Handle input data - can be single DataArray or tuple/list
        if isinstance(current_data, (list, tuple)):
            fill_data = current_data[0]
            contour_data = current_data[1] if len(current_data) > 1 else None
        else:
            fill_data = current_data
            contour_data = None
        
        # Set global extent
        ax.set_global()
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black')
        
        # Add boundaries based on user selection
        current_boundaries = regional_boundary_list[i]
        if current_boundaries is True or current_boundaries == 'countries':
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.7)
        elif current_boundaries == 'ar6':
            ar6_regions = regionmask.defined_regions.ar6.land
            ar6_regions.plot_regions(ax=ax, 
                                   line_kws={'linewidth': 0.6, 'color': 'black', 'alpha': 0.4},
                                   add_label=False)
        
        # Handle fill data coordinates
        if 'latitude' in fill_data.coords:
            fill_data = fill_data.rename({'latitude': 'lat'})
        if 'longitude' in fill_data.coords:
            fill_data = fill_data.rename({'longitude': 'lon'})
        
        # Use extends parameter directly
        extend_param = extend_list[i]
        
        # Set color limits
        vmin = vmin_list[i]
        vmax = vmax_list[i]
        if vmin is None or vmax is None:
            data_range = np.nanmax(np.abs(fill_data.values))
            if vmin is None:
                vmin = -data_range
            if vmax is None:
                vmax = data_range
        
        # Create filled contour plot
        if colorbar_level_list[i] is not None:
            im = ax.contourf(fill_data.lon, fill_data.lat, fill_data.values, 
                           levels=colorbar_level_list[i],
                           transform=ccrs.PlateCarree(),
                           cmap=colormap_list[i],
                           vmin=vmin, vmax=vmax,
                           extend=extend_param)
        else:
            im = ax.contourf(fill_data.lon, fill_data.lat, fill_data.values,
                           transform=ccrs.PlateCarree(),
                           cmap=colormap_list[i],
                           vmin=vmin, vmax=vmax,
                           extend=extend_param)
        
        # Add contour lines if contour data is provided
        if contour_data is not None:
            if 'latitude' in contour_data.coords:
                contour_data = contour_data.rename({'latitude': 'lat'})
            if 'longitude' in contour_data.coords:
                contour_data = contour_data.rename({'longitude': 'lon'})
            
            if contour_level_list[i] is not None:
                cs = ax.contour(contour_data.lon, contour_data.lat, contour_data.values,
                               levels=contour_level_list[i],
                               transform=ccrs.PlateCarree(),
                               colors='black', linewidths=0.6, alpha=0.7)
            else:
                cs = ax.contour(contour_data.lon, contour_data.lat, contour_data.values,
                               transform=ccrs.PlateCarree(),
                               colors='black', linewidths=0.6, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%g')
        
        # Add hatching
        if hatching_list[i] is not None and hatching_data_list[i] is not None:
            hatch_data = hatching_data_list[i]
            if 'latitude' in hatch_data.coords:
                hatch_data = hatch_data.rename({'latitude': 'lat'})
            if 'longitude' in hatch_data.coords:
                hatch_data = hatch_data.rename({'longitude': 'lon'})
            
            hatching_data_valid = ~np.isnan(hatch_data.values)
            hatch_mask = hatching_data_valid
            
            if np.any(hatch_mask):
                X, Y = np.meshgrid(hatch_data.lon, hatch_data.lat)
                if hatching_style_list[i] == 'overlay':
                    ax.contourf(X, Y, hatch_mask.astype(int), 
                               levels=[0.5, 1.5], colors='none', 
                               hatches=[hatching_list[i]], 
                               transform=ccrs.PlateCarree())
                elif hatching_style_list[i] == 'white':
                    ax.contourf(X, Y, hatch_mask.astype(int),
                               levels=[0.5, 1.5], colors=['white'],
                               hatches=[hatching_list[i]],
                               transform=ccrs.PlateCarree(),
                               alpha=1.0, zorder=5)
        
        # Add RAMIP regions (copied from original function)
        current_ramip = ramip_regions_list[i]
        if current_ramip:
            # RAMIP region definitions
            regions_coords = {
                'east_asia': {'lon_west': 95, 'lon_east': 133, 'lat_bot': 20, 'lat_top': 53},
                'south_asia': {'lon_west': 65, 'lon_east': 95, 'lat_bot': 5, 'lat_top': 35},
                'africa_mideast': {'lon_west': 340, 'lon_east': 60, 'lat_bot': -35, 'lat_top': 35},
                'north_america': {'lon_west': 210, 'lon_east': 315, 'lat_bot': 25, 'lat_top': 70},
                'europe': {'lon_west': 340, 'lon_east': 45, 'lat_bot': 35, 'lat_top': 70}
            }
            
            # Determine regions to show
            regions_to_show = []
            if current_ramip is True:
                regions_to_show = ['east_asia', 'south_asia', 'africa_mideast', 'north_america', 'europe']
            elif isinstance(current_ramip, str):
                if current_ramip == 'north_america_europe':
                    regions_to_show = ['north_america', 'europe']
                else:
                    regions_to_show = [current_ramip]
            elif isinstance(current_ramip, (list, tuple)):
                for region in current_ramip:
                    if region == 'north_america_europe':
                        regions_to_show.extend(['north_america', 'europe'])
                    else:
                        regions_to_show.append(region)
            
            # Draw regions
            for region in regions_to_show:
                if region in regions_coords:
                    coords = regions_coords[region]
                    if region in ['africa_mideast', 'europe']:  # Handle dateline crossing
                        width = coords['lon_east'] + (360 - coords['lon_west'])
                    else:
                        width = coords['lon_east'] - coords['lon_west']
                    
                    ax.add_patch(mpatches.Rectangle(
                        xy=[coords['lon_west'], coords['lat_bot']], 
                        width=width,
                        height=(coords['lat_top'] - coords['lat_bot']),
                        facecolor='none', linestyle='--', linewidth=1.5,
                        edgecolor='black', alpha=1,
                        transform=ccrs.PlateCarree()))
        
        # Add gridlines
        if show_gridlines_list[i]:
            ax.gridlines(draw_labels=False, linestyle='-', alpha=0.3, color='gray')
        
        # Create individual colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.08, shrink=0.8, aspect=15,
                           extend=extend_param)
        cbar.set_label(colorbar_title_list[i], fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        # Add subplot title
        ax.set_title(title_list[i], fontsize=12, pad=10)
        
        # Add textbox if provided
        if textbox_text_list[i] is not None:
            bbox_props = dict(boxstyle="round,pad=0.2", facecolor='white', 
                            edgecolor='black', linewidth=0.8)
            ax.text(0.02, 0.02, textbox_text_list[i], transform=ax.transAxes,
                   fontsize=10, verticalalignment='center', horizontalalignment='left',
                   bbox=bbox_props, zorder=10)
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, fontweight='regular', y=0.95)
    
    # Hide any unused subplots
    total_subplots = rows * cols
    for i in range(n_plots, total_subplots):
        ax_unused = fig.add_subplot(rows, cols, i + 1)
        ax_unused.set_visible(False)
    
    # Adjust layout
    plt.subplots_adjust(**subplot_spacing)
    
    return fig, axes

# (9) Apply significance and land masks
def apply_masks(data, 
                get_significance=False, 
                agreement_threshold=0.0,
                threshold_type='minimum',
                agreement_dim='auto',
                get_land_mask=True,
                baseline_data=None,
                zero_handling='exclude',
                zero_threshold=1e-10):
    """
    Consolidated function for applying significance and land masks with zero handling
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input data to be masked
    get_significance : bool, optional
        Whether to apply significance masking based on ensemble agreement
    agreement_threshold : float, optional
        Agreement threshold for significance (0-1)
    threshold_type : str, optional
        Type of threshold to apply ('minimum' or 'maximum')
    agreement_dim : str, optional
        Dimension to calculate agreement over ('models', 'members', or 'auto')
    get_land_mask : bool, optional
        Whether to apply land masking
    baseline_data : xarray.DataArray, optional
        Baseline data for significance testing (required if get_significance=True)
    zero_handling : str, optional
        How to handle zero values ('exclude', 'neutral', 'positive', 'negative', 'magnitude')
        - 'exclude': Ignore zeros (original behavior)
        - 'neutral': Count zeros as separate category, require agreement among non-zeros
        - 'positive': Treat zeros as positive values
        - 'negative': Treat zeros as negative values  
        - 'magnitude': Use magnitude-based agreement instead of sign-based
    zero_threshold : float, optional
        Values with absolute value <= this threshold are considered "zero" (default: 1e-10)
        
    Returns:
    --------
    tuple : (xarray.DataArray, xarray.Dataset or None)
        Masked data array and land mask dataset (if land_masking=True)
    """
    masked_data = data.copy()
    da_mask = None
    
    # Apply significance mask if requested
    if get_significance and (agreement_threshold > 0.0 or threshold_type == 'maximum'):
        if baseline_data is None:
            raise ValueError("baseline_data required for significance masking")
        
        # Validate parameters
        if threshold_type not in ['minimum', 'maximum']:
            raise ValueError("threshold_type must be 'minimum' or 'maximum'")
            
        if zero_handling not in ['exclude', 'neutral', 'positive', 'negative', 'magnitude']:
            raise ValueError("zero_handling must be one of: 'exclude', 'neutral', 'positive', 'negative', 'magnitude'")
        
        # Determine which dimension to use for agreement calculation
        if agreement_dim == 'auto':
            if "member" in baseline_data.dims:
                dim_name = "member"
            elif "model" in baseline_data.dims:
                dim_name = "model"
            else:
                raise ValueError("No 'member' or 'model' dimension found in baseline_data")
        elif agreement_dim == 'members':
            dim_name = "member"
        elif agreement_dim == 'models':
            dim_name = "model"
        else:
            raise ValueError("agreement_dim must be 'auto', 'members', or 'models'")
            
        if dim_name not in baseline_data.dims:
            raise ValueError(f"Specified dimension '{dim_name}' not found in baseline_data")
        
        # Calculate agreement based on zero handling method
        if zero_handling == 'magnitude':
            # Magnitude-based agreement: agreement on whether change is "large" or "small"
            magnitude_threshold = np.percentile(np.abs(baseline_data.values), 50)  # Use median as threshold
            large_change = np.abs(baseline_data) > magnitude_threshold
            large_counts = large_change.sum(dim=dim_name)
            small_counts = (~large_change).sum(dim=dim_name)
            total_members = len(baseline_data[dim_name])
            
            agreement_fraction = xr.where(
                large_counts > small_counts,
                large_counts / total_members,
                small_counts / total_members
            )
            
        else:
            # Sign-based agreement with different zero handling
            # Identify zeros using threshold
            is_zero = np.abs(baseline_data) <= zero_threshold
            diff_signs = np.sign(baseline_data)
            
            if zero_handling == 'positive':
                # Treat zeros as positive
                diff_signs = xr.where(is_zero, 1, diff_signs)
            elif zero_handling == 'negative':
                # Treat zeros as negative  
                diff_signs = xr.where(is_zero, -1, diff_signs)
            elif zero_handling == 'neutral':
                # Keep zeros as 0, but handle agreement calculation differently
                pass  # diff_signs already has zeros as 0
            # For 'exclude': zeros remain as 0 and get excluded (original behavior)
            
            # Count agreement
            positive_counts = xr.where(diff_signs > 0, 1, 0).sum(dim=dim_name)
            negative_counts = xr.where(diff_signs < 0, 1, 0).sum(dim=dim_name)
            zero_counts = xr.where(diff_signs == 0, 1, 0).sum(dim=dim_name)
            
            if zero_handling == 'neutral':
                # For neutral handling, require agreement among non-zero values
                # Zeros don't contribute to agreement but don't reduce total either
                non_zero_members = positive_counts + negative_counts
                agreement_fraction = xr.where(
                    non_zero_members > 0,  # Avoid division by zero
                    xr.where(
                        positive_counts > negative_counts,
                        positive_counts / non_zero_members,
                        negative_counts / non_zero_members
                    ),
                    0.5  # If all zeros, set to 50% agreement (neutral)
                )
            else:
                # Standard agreement calculation (zeros treated as assigned or excluded)
                total_members = len(baseline_data[dim_name])
                if zero_handling == 'exclude':
                    # Adjust total to exclude zeros
                    total_members_adjusted = positive_counts + negative_counts
                    agreement_fraction = xr.where(
                        total_members_adjusted > 0,  # Avoid division by zero
                        xr.where(
                            positive_counts > negative_counts,
                            positive_counts / total_members_adjusted,
                            negative_counts / total_members_adjusted
                        ),
                        0.5  # If no non-zero values, set to 50%
                    )
                else:
                    # Use full total (zeros counted as positive or negative)
                    agreement_fraction = xr.where(
                        positive_counts > negative_counts,
                        positive_counts / total_members,
                        negative_counts / total_members
                    )
        
        # Apply appropriate mask based on threshold type
        if threshold_type == 'maximum':
            masked_data = masked_data.where(agreement_fraction <= agreement_threshold)
        else:  # threshold_type == 'minimum'
            masked_data = masked_data.where(agreement_fraction >= agreement_threshold)
    
    # Apply land mask if requested
    if get_land_mask:
        # Read land mask file
        da_mask = xr.open_dataset("/home/persad_users/vsu66/Research/EDF/b.e21.BSSP370cmip6.f09_g17.LE2-1231.001.clm2.h3.PCT_NAT_PFT.20950101-21001231.nc")
        da_mask = da_mask.interp(coords={"lat":masked_data.lat.values, "lon":masked_data.lon.values})
        
        # Apply land and non-barren masks
        masked_data = masked_data.where(
            (da_mask['landmask'].values == 1) & 
            (da_mask['PCT_NAT_PFT'].isel(time=-1).sel(natpft=0).values <= 80)
        )
    
    return masked_data, da_mask


# (10) Calculating the regional effects of GHG and aerosol reduction relative to historical
def table_regional_time_mean(historical_var, ssp370_var, global_var, ssp126_var,\
                             start_analysis,\
                                end_analysis,\
                                    lat_bot = -90,\
                                        lat_top = 90,\
                                            lon_west = 0,\
                                                lon_east = 360):
    
    # (1) Take the period mean
    
    historical_regional_time_mean = regional_time_mean(historical_var,\
                                                       start_analysis = cftime.DatetimeNoLeap(1961, 1, 1, 12, 0, 0, 0, has_year_zero=True),\
                                                        end_analysis = cftime.DatetimeNoLeap(1990, 12, 31, 0, 0, 0, 0, has_year_zero=True),\
                                                            lat_bot = lat_bot,\
                                                                lat_top = lat_top,\
                                                                    lon_west = lon_west,\
                                                                        lon_east = lon_east)
    
    ssp370_regional_time_mean = regional_time_mean(ssp370_var,\
                                               start_analysis = start_analysis,\
                                                        end_analysis = end_analysis,\
                                                            lat_bot = lat_bot,\
                                                                lat_top = lat_top,\
                                                                    lon_west = lon_west,\
                                                                        lon_east = lon_east)
    
    global_regional_time_mean = regional_time_mean(global_var,\
                                                   start_analysis = start_analysis,\
                                                    end_analysis = end_analysis,\
                                                        lat_bot = lat_bot,\
                                                            lat_top = lat_top,\
                                                                lon_west = lon_west,\
                                                                    lon_east = lon_east)
    
    ssp126_regional_time_mean = regional_time_mean(ssp126_var,\
                                                   start_analysis = start_analysis,\
                                                    end_analysis = end_analysis,\
                                                        lat_bot = lat_bot,\
                                                            lat_top = lat_top,\
                                                                lon_west = lon_west,\
                                                                    lon_east = lon_east)
    
    # (2) Get ensemble *anomalies* from baseline
    if "member" in historical_regional_time_mean.dims:
        ssp370_var_anomaly = ssp370_regional_time_mean.mean(dim='member') - historical_regional_time_mean.mean(dim='member')
        ssp126_var_anomaly = ssp126_regional_time_mean.mean(dim='member') - historical_regional_time_mean.mean(dim='member')
        ghg_eff = ssp126_regional_time_mean.mean(dim='member') - global_regional_time_mean.mean(dim='member')
        aerosol_eff = global_regional_time_mean.mean(dim='member') - ssp370_regional_time_mean.mean(dim='member')
    elif "model" in historical_regional_time_mean.dims:
        ssp370_var_anomaly = ssp370_regional_time_mean.mean(dim='model') - historical_regional_time_mean.mean(dim='model')
        ssp126_var_anomaly = ssp126_regional_time_mean.mean(dim='model') - historical_regional_time_mean.mean(dim='model')
        ghg_eff = ssp126_regional_time_mean.mean(dim='model') - global_regional_time_mean.mean(dim='model')
        aerosol_eff = global_regional_time_mean.mean(dim='model') - ssp370_regional_time_mean.mean(dim='model')
        

    # (3) Print values

    print(f"Annual high fire danger days in historical: {historical_regional_time_mean.values.item(0):.2f} days")
    print(f"Change in regional annual high fire danger days in ssp3-7.0 relative to historical: {ssp370_var_anomaly.values.item(0):.2f} days")
    print(f"Change in regional annual high fire danger days in ssp1-2.6 relative to historical: {ssp126_var_anomaly.values.item(0):.2f} days")
    print(f"Change in regional annual high fire danger days due to GHG reduction: {ghg_eff.values.item(0):.2f} days")
    print(f"Change in regional annual high fire danger days due to aerosol reduction: {aerosol_eff.values.item(0):.2f} days")

# (11) Calculating regional mean 
def regional_time_mean(da,\
                       start_analysis,\
                        end_analysis,\
                            lat_bot = -90,\
                                lat_top = 90,\
                                    lon_west = 0,\
                                        lon_east = 360):
     # (0) Modify coordinate system for regions that cross the zero longitude

    if lon_west > lon_east:

        # To do computation on the Mediterranean (crossing the zero longitude), we need
        # to change the lons from 0-360 to -180 to 180 and set the lon_wests[i] value
        # appropriately (we don't change lon_easts[i] because we don't change lon: 0 to 180
        # we're changing lon: 180-360 to lon: -180 to 0
        # !! In the future: may need a way that can accommodate when lon_easts[i] >= 180

        lon_west = lon_west - 360

        # Turn longitude values of 180, ... to -180, ... (but longitude values of 0, ... stays)
        # then sort

        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby('lon')

    # (1) Take the period mean

    da = da.sel(time=slice(start_analysis, end_analysis)).sel(lat=slice(lat_bot, lat_top), lon=slice(lon_west, lon_east))
    da = weighted_horizontal_avg(da, member = False, time = True)

    return da
