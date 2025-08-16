import numpy as np
import pandas as pd 
import xarray as xr
import cartopy.util as cutil
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cftime
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

# (7) Create map
def create_map(data,
               title,
               label,
               levels,
               cmap='coolwarm',
               baseline_data=None,
               contour_data=None,
               min_agreement=0.0,
               land_masking=True,
               region_boxes=False,
               contour=False,
               show_stippling=False,
               textbox_outside=False,
               map_proj=ccrs.Robinson()):
    """
    Consolidated function for creating maps
    
    Parameters:
    -----------
    data : xarray.DataArray
        Data to plot
    title : str
        Plot title
    label : str
        Colorbar label
    levels : array-like
        Contour levels for plotting
    cmap : str, optional
        Colormap name
    baseline_data : xarray.DataArray, optional
        Baseline data for difference plots
    contour_data : xarray.DataArray, optional
        Data to use for contour lines (if different from baseline_data)
    min_agreement : float, optional
        Minimum agreement for significance masking
    land_masking : bool, optional
        Whether to apply land masking
    region_boxes : bool, optional
        Whether to add region boxes
    contour : bool, optional
        Whether to add contour lines
    show_stippling : bool, optional
        Whether to show stippling in areas that pass land masking but not significance
    textbox_outside : bool, optional
        Whether to place textbox outside the map
    map_proj : cartopy.crs, optional
        Map projection
        
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    # Make data cyclic
    new_data, new_lon = cutil.add_cyclic(data, x=data.lon)
    da_map_cyclic = xr.DataArray(new_data,
                                coords={'lat': data.lat, 'lon': new_lon},
                                dims=["lat", "lon"])
    
    # First apply only land masking to get land-only data
    da_land_masked, da_mask = apply_masks(
        da_map_cyclic,
        get_significance=False,
        get_land_mask=land_masking
    )
    
    # Then apply both masks to get final plot data
    da_map_masked, _ = apply_masks(
        da_map_cyclic,
        get_significance=(baseline_data is not None),
        min_agreement=min_agreement,
        get_land_mask=land_masking,
        baseline_data=baseline_data
    )
    
    # Set colorbar and textbox positions
    if textbox_outside:
        cb_pad = 0.1
        text_x = 0.015
        text_y = -0.58
    else:
        cb_pad = 0.05
        text_x = 0.015
        text_y = 0.05
    
    # Create figure
    fig = plt.figure(figsize=[10, 7.5])
    ax = fig.add_subplot(111, projection=map_proj)
    
    # Create main plot
    da_map_masked.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend="both",
        levels=levels,
        cbar_kwargs={
            'label': label,
            'orientation': 'horizontal',
            'pad': cb_pad
        }
    )
    
    # Add contours if requested
    if contour:
        # Determine which data to use for contours
        if contour_data is not None:
            plot_contours = contour_data
        elif baseline_data is not None:
            plot_contours = baseline_data
        else:
            plot_contours = data
            
        # Make contour data cyclic and mask
        new_contour_data, new_contour_lon = cutil.add_cyclic(plot_contours, x=plot_contours.lon)
        contour_cyclic = xr.DataArray(new_contour_data,
                                    coords={'lat': plot_contours.lat, 'lon': new_contour_lon},
                                    dims=["lat", "lon"])
        contour_masked, _ = apply_masks(contour_cyclic, get_land_mask=land_masking)
        
        # Plot contours
        contour = contour_masked.plot.contour(
            ax=ax,
            transform=ccrs.PlateCarree(),
            colors='black'
        )
        ax.clabel(contour, inline=True, fontsize=8)
    
    # Add stippling for areas that pass land mask but not significance
    if show_stippling and baseline_data is not None and min_agreement > 0:
        # Create a mask for stippling (True where we want stipples)
        stipple_mask = xr.where(
            xr.isnan(da_map_masked) & ~xr.isnan(da_land_masked),
            True,
            False
        )
        
        # Convert to numpy arrays for plotting
        lons, lats = np.meshgrid(new_lon, data.lat)
        
        # Plot stippling
        ax.scatter(
            lons[stipple_mask],
            lats[stipple_mask],
            color='black',
            s=1,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            marker='//'
        )
    
    # Add region boxes if requested
    if region_boxes:
        add_region_boxes(ax, title)
    
    # Calculate and display weighted average
    if da_mask is not None:
        value = weighted_horizontal_avg(
            da_map_masked.where(
                (da_mask['landmask'].values == 1) &
                (da_mask['PCT_NAT_PFT'].isel(time=-1).sel(natpft=0).values <= 80),
                drop=False
            ),
            ensemble=False,
            time=False
        )
    else:
        value = weighted_horizontal_avg(
            da_map_masked,
            ensemble=False,
            time=False
        )
    
    # Add textbox with value
    ax.text(
        text_x, text_y,
        f"{value.values.item():.2f}",
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 8},
        transform=ax.transAxes
    )
    
    # Finalize plot
    ax.set_title(title)
    ax.coastlines()
    ax.gridlines(color='grey', crs=ccrs.PlateCarree(), alpha=0.25)
    
    return fig, ax

# (8) Add region boxes
def add_region_boxes(ax, title):
    """
    Adds regional boundary boxes to the map based on the title.
    
    Parameters:
    -----------
    ax : matplotlib.axes
        The axis to add the boxes to
    title : str
        Title of the plot that determines which boxes to draw
    """
    if title == 'Effect of Aerosol Emission Reduction':
        boxes = [
            ('nam', lon_west_nam, lat_bot_nam, lon_east_nam-lon_west_nam, lat_top_nam-lat_bot_nam),
            ('eur', lon_west_eur, lat_bot_eur, lon_east_eur+(360-lon_west_eur), lat_top_eur-lat_bot_eur),
            ('afr', lon_west_afr, lat_bot_afr, lon_east_afr+(360-lon_west_afr), lat_top_afr-lat_bot_afr),
            ('eas', lon_west_eas, lat_bot_eas, lon_east_eas-lon_west_eas, lat_top_eas-lat_bot_eas),
            ('sas', lon_west_sas, lat_bot_sas, lon_east_sas-lon_west_sas, lat_top_sas-lat_bot_sas)
        ]
    elif title == 'Effect of East Asian Aerosol Reduction':
        boxes = [('eas', lon_west_eas, lat_bot_eas, lon_east_eas-lon_west_eas, lat_top_eas-lat_bot_eas)]
    elif title == 'Effect of South Asian Aerosol Reduction':
        boxes = [('sas', lon_west_sas, lat_bot_sas, lon_east_sas-lon_west_sas, lat_top_sas-lat_bot_sas)]
    elif title == 'Effect of African & Mid-Eastern Aerosol Reduction':
        boxes = [('afr', lon_west_afr, lat_bot_afr, lon_east_afr+(360-lon_west_afr), lat_top_afr-lat_bot_afr)]
    elif title == 'Effect of North American & European Aerosol Reduction':
        boxes = [
            ('nam', lon_west_nam, lat_bot_nam, lon_east_nam-lon_west_nam, lat_top_nam-lat_bot_nam),
            ('eur', lon_west_eur, lat_bot_eur, lon_east_eur+(360-lon_west_eur), lat_top_eur-lat_bot_eur)
        ]
    else:
        return

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


# (10)Calculating the regional effects of GHG and aerosol reduction relative to historical
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

# (11)Calculating regional mean 
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
