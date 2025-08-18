#!/usr/bin/env python3
"""
Annual 95th Percentile Exceedances Analysis
Extracted from fwi_analysis_percentile.ipynb
This script performs annual analysis of 95th percentile exceedances and generates plots.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import utility functions
from ramip_fwi_utilities import (
    read_zarr, 
    apply_masks, 
    weighted_horizontal_avg, 
    create_global_map, 
    create_global_map_grid
)

def main():
    """Main function to perform annual 95th percentile exceedances analysis."""
    
    print("Starting Annual 95th Percentile Exceedances Analysis...")
    
    # Create output directory for plots
    output_dir = Path("annual_analysis_plots")
    output_dir.mkdir(exist_ok=True)
    
    # =============================================================================
    # DATA LOADING - Historical
    # =============================================================================
    print("Loading historical data...")
    
    # Load historical FWI data for each model
    noresm2_historical_fwi = read_zarr('NorESM2-LM', 'historical', 'FWI',
                                       start_year=1961, end_year=1990)
    spear_historical_fwi = read_zarr('SPEAR', 'historical', 'FWI',
                                     start_year=1961, end_year=1990)
    mri_historical_fwi = read_zarr('MRI-ESM2-0', 'historical', 'FWI',
                                   start_year=1961, end_year=1990)
    
    # Calculate 95th percentile threshold for historical data
    print("Calculating historical 95th percentile thresholds...")
    noresm2_historical_fwi_95th_threshold = noresm2_historical_fwi.quantile(0.95, dim=['time', 'member'])
    spear_historical_fwi_95th_threshold = spear_historical_fwi.quantile(0.95, dim=['time', 'member'])
    mri_historical_fwi_95th_threshold = mri_historical_fwi.quantile(0.95, dim=['time', 'member'])
    
    # Calculate 95th percentile exceedances for historical data
    print("Calculating historical 95th percentile exceedances...")
    noresm2_historical_fwi_95th_boolean = noresm2_historical_fwi > noresm2_historical_fwi_95th_threshold
    spear_historical_fwi_95th_boolean = spear_historical_fwi > spear_historical_fwi_95th_threshold
    mri_historical_fwi_95th_boolean = mri_historical_fwi > mri_historical_fwi_95th_threshold
    
    # Calculate annual counts for historical data
    noresm2_historical_fwi_95th_annual_count = noresm2_historical_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_historical_fwi_95th_annual_count = spear_historical_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_historical_fwi_95th_annual_count = mri_historical_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    # Combine models
    multi_historical_fwi_95th_annual_count = xr.concat([
        noresm2_historical_fwi_95th_annual_count, 
        spear_historical_fwi_95th_annual_count, 
        mri_historical_fwi_95th_annual_count
    ], dim="model")
    
    historical_fwi_95th_annual_count = multi_historical_fwi_annual_count.compute()
    print("Historical 95th percentile annual count is ready")
    
    # =============================================================================
    # DATA LOADING - Future Scenarios
    # =============================================================================
    print("Loading future scenario data...")
    
    # SSP3-7.0
    noresm2_ssp370_fwi = read_zarr('NorESM2-LM', 'ssp370', 'FWI',
                                   start_year=2041, end_year=2050)
    spear_ssp370_fwi = read_zarr('SPEAR', 'ssp370', 'FWI',
                                 start_year=2041, end_year=2050)
    mri_ssp370_fwi = read_zarr('MRI-ESM2-0', 'ssp370', 'FWI',
                               start_year=2041, end_year=2050)
    
    # Global aerosol reduction
    noresm2_global_fwi = read_zarr('NorESM2-LM', 'global', 'FWI',
                                   start_year=2041, end_year=2050)
    spear_global_fwi = read_zarr('SPEAR', 'global', 'FWI',
                                 start_year=2041, end_year=2050)
    mri_global_fwi = read_zarr('MRI-ESM2-0', 'global', 'FWI',
                               start_year=2041, end_year=2050)
    
    # SSP1-2.6
    noresm2_ssp126_fwi = read_zarr('NorESM2-LM', 'ssp126', 'FWI',
                                   start_year=2041, end_year=2050)
    spear_ssp126_fwi = read_zarr('SPEAR', 'ssp126', 'FWI',
                                 start_year=2041, end_year=2050)
    mri_ssp126_fwi = read_zarr('MRI-ESM2-0', 'ssp126', 'FWI',
                               start_year=2041, end_year=2050)
    
    # Regional aerosol reductions
    noresm2_eas_fwi = read_zarr('NorESM2-LM', 'eas', 'FWI',
                                start_year=2041, end_year=2050)
    spear_eas_fwi = read_zarr('SPEAR', 'eas', 'FWI',
                              start_year=2041, end_year=2050)
    mri_eas_fwi = read_zarr('MRI-ESM2-0', 'eas', 'FWI',
                            start_year=2041, end_year=2050)
    
    noresm2_sas_fwi = read_zarr('NorESM2-LM', 'sas', 'FWI',
                                start_year=2041, end_year=2050)
    spear_sas_fwi = read_zarr('SPEAR', 'sas', 'FWI',
                              start_year=2041, end_year=2050)
    mri_sas_fwi = read_zarr('MRI-ESM2-0', 'sas', 'FWI',
                            start_year=2041, end_year=2050)
    
    noresm2_afr_fwi = read_zarr('NorESM2-LM', 'afr', 'FWI',
                                start_year=2041, end_year=2050)
    spear_afr_fwi = read_zarr('SPEAR', 'afr', 'FWI',
                              start_year=2041, end_year=2050)
    mri_afr_fwi = read_zarr('MRI-ESM2-0', 'afr', 'FWI',
                            start_year=2041, end_year=2050)
    
    noresm2_nae_fwi = read_zarr('NorESM2-LM', 'nae', 'FWI',
                                start_year=2041, end_year=2050)
    spear_nae_fwi = read_zarr('SPEAR', 'nae', 'FWI',
                              start_year=2041, end_year=2050)
    mri_nae_fwi = read_zarr('MRI-ESM2-0', 'nae', 'FWI',
                            start_year=2041, end_year=2050)
    
    # Calculate 95th percentile exceedances for future scenarios
    print("Calculating future scenario 95th percentile exceedances...")
    
    # SSP3-7.0
    noresm2_ssp370_fwi_95th_boolean = noresm2_ssp370_fwi > noresm2_historical_fwi_95th_threshold
    spear_ssp370_fwi_95th_boolean = spear_ssp370_fwi > spear_historical_fwi_95th_threshold
    mri_ssp370_fwi_95th_boolean = mri_ssp370_fwi > mri_historical_fwi_95th_threshold
    
    # Global
    noresm2_global_fwi_95th_boolean = noresm2_global_fwi > noresm2_historical_fwi_95th_threshold
    spear_global_fwi_95th_boolean = spear_global_fwi > spear_historical_fwi_95th_threshold
    mri_global_fwi_95th_boolean = mri_global_fwi > mri_historical_fwi_95th_threshold
    
    # SSP1-2.6
    noresm2_ssp126_fwi_95th_boolean = noresm2_ssp126_fwi > noresm2_historical_fwi_95th_threshold
    spear_ssp126_fwi_95th_boolean = spear_ssp126_fwi > spear_historical_fwi_95th_threshold
    mri_ssp126_fwi_95th_boolean = mri_ssp126_fwi > mri_historical_fwi_95th_threshold
    
    # Regional scenarios
    noresm2_eas_fwi_95th_boolean = noresm2_eas_fwi > noresm2_historical_fwi_95th_threshold
    spear_eas_fwi_95th_boolean = spear_eas_fwi > spear_historical_fwi_95th_threshold
    mri_eas_fwi_95th_boolean = mri_eas_fwi > mri_historical_fwi_95th_threshold
    
    noresm2_sas_fwi_95th_boolean = noresm2_sas_fwi > noresm2_historical_fwi_95th_threshold
    spear_sas_fwi_95th_boolean = spear_sas_fwi > spear_historical_fwi_95th_threshold
    mri_sas_fwi_95th_boolean = mri_sas_fwi > mri_historical_fwi_95th_threshold
    
    noresm2_afr_fwi_95th_boolean = noresm2_afr_fwi > noresm2_historical_fwi_95th_threshold
    spear_afr_fwi_95th_boolean = spear_afr_fwi > spear_historical_fwi_95th_threshold
    mri_afr_fwi_95th_boolean = mri_afr_fwi > mri_historical_fwi_95th_threshold
    
    noresm2_nae_fwi_95th_boolean = noresm2_nae_fwi > noresm2_historical_fwi_95th_threshold
    spear_nae_fwi_95th_boolean = spear_nae_fwi > spear_historical_fwi_95th_threshold
    mri_nae_fwi_95th_boolean = mri_nae_fwi > mri_historical_fwi_95th_threshold
    
    # Calculate annual counts for future scenarios
    # SSP3-7.0
    noresm2_ssp370_fwi_95th_annual_count = noresm2_ssp370_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_ssp370_fwi_95th_annual_count = spear_ssp370_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_ssp370_fwi_95th_annual_count = mri_ssp370_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    # Global
    noresm2_global_fwi_95th_annual_count = noresm2_global_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_global_fwi_95th_annual_count = spear_global_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_global_fwi_95th_annual_count = mri_global_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    # SSP1-2.6
    noresm2_ssp126_fwi_95th_annual_count = noresm2_ssp126_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_ssp126_fwi_95th_annual_count = spear_ssp126_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_ssp126_fwi_95th_annual_count = mri_ssp126_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    # Regional scenarios
    noresm2_eas_fwi_95th_annual_count = noresm2_eas_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_eas_fwi_95th_annual_count = spear_eas_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_eas_fwi_95th_annual_count = mri_eas_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    noresm2_sas_fwi_95th_annual_count = noresm2_sas_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_sas_fwi_95th_annual_count = spear_sas_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_sas_fwi_95th_annual_count = mri_sas_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    noresm2_afr_fwi_95th_annual_count = noresm2_afr_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_afr_fwi_95th_annual_count = spear_afr_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_afr_fwi_95th_annual_count = mri_afr_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    noresm2_nae_fwi_95th_annual_count = noresm2_nae_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    spear_nae_fwi_95th_annual_count = spear_nae_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    mri_nae_fwi_95th_annual_count = mri_nae_fwi_95th_boolean.resample(time='Y').sum(skipna=True).mean(dim=['time','member'])
    
    # Combine models for each scenario
    multi_ssp370_fwi_95th_annual_count = xr.concat([
        noresm2_ssp370_fwi_95th_annual_count, spear_ssp370_fwi_95th_annual_count, mri_ssp370_fwi_95th_annual_count
    ], dim="model")
    ssp370_fwi_95th_annual_count = multi_ssp370_fwi_95th_annual_count.compute()
    print("SSP3-7.0 95th percentile annual count is ready")
    
    multi_global_fwi_95th_annual_count = xr.concat([
        noresm2_global_fwi_95th_annual_count, spear_global_fwi_95th_annual_count, mri_global_fwi_95th_annual_count
    ], dim="model")
    global_fwi_95th_annual_count = multi_global_fwi_95th_annual_count.compute()
    print("Global aerosol reduction 95th percentile annual count is ready")
    
    multi_ssp126_fwi_95th_annual_count = xr.concat([
        noresm2_ssp126_fwi_95th_annual_count, spear_ssp126_fwi_95th_annual_count, mri_ssp126_fwi_95th_annual_count
    ], dim="model")
    ssp126_fwi_95th_annual_count = multi_ssp126_fwi_95th_annual_count.compute()
    print("SSP1-2.6 95th percentile annual count is ready")
    
    multi_eas_fwi_95th_annual_count = xr.concat([
        noresm2_eas_fwi_95th_annual_count, spear_eas_fwi_95th_annual_count, mri_eas_fwi_95th_annual_count
    ], dim="model")
    eas_fwi_95th_annual_count = multi_eas_fwi_95th_annual_count.compute()
    print("East Asia aerosol reduction 95th percentile annual count is ready")
    
    multi_sas_fwi_95th_annual_count = xr.concat([
        noresm2_sas_fwi_95th_annual_count, spear_sas_fwi_95th_annual_count, mri_sas_fwi_95th_annual_count
    ], dim="model")
    sas_fwi_95th_annual_count = multi_sas_fwi_95th_annual_count.compute()
    print("South Asia aerosol reduction 95th percentile annual count is ready")
    
    multi_afr_fwi_95th_annual_count = xr.concat([
        noresm2_afr_fwi_95th_annual_count, spear_afr_fwi_95th_annual_count, mri_afr_fwi_95th_annual_count
    ], dim="model")
    afr_fwi_95th_annual_count = multi_afr_fwi_95th_annual_count.compute()
    print("Africa & Middle East aerosol reduction 95th percentile annual count is ready")
    
    multi_nae_fwi_95th_annual_count = xr.concat([
        noresm2_nae_fwi_95th_annual_count, spear_nae_fwi_95th_annual_count, mri_nae_fwi_95th_annual_count
    ], dim="model")
    nae_fwi_95th_annual_count = multi_nae_fwi_95th_annual_count.compute()
    print("North America & Europe aerosol reduction 95th percentile annual count is ready")
    
    # =============================================================================
    # APPLY MASKS AND CALCULATE GLOBAL AVERAGES
    # =============================================================================
    print("Applying masks and calculating global averages...")
    
    # Apply masks for historical data
    historical_fwi_95th_annual_count_masked, _ = apply_masks(historical_fwi_95th_annual_count, 
                                                             get_significance=False,
                                                             get_land_mask=True)
    
    # Calculate global average for historical
    historical_fwi_95th_annual_count_masked_globalavg = weighted_horizontal_avg(
        historical_fwi_95th_annual_count_masked.mean('model'), 
        ensemble=False, 
        time=False
    )
    
    # =============================================================================
    # PLOT HISTORICAL DATA
    # =============================================================================
    print("Creating historical plot...")
    
    fig, ax = create_global_map(
        historical_fwi_95th_annual_count_masked.mean('model'), 
        title="Historical 95th Percentile Exceedances (1961-1990)",
        colormap='Reds',
        colorbar_title="Days per Year",
        textbox_text=f"{historical_fwi_95th_annual_count_masked_globalavg.values.item():.1f}"
    )
    
    # Save historical plot
    plt.savefig(output_dir / "historical_95th_percentile.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Historical plot saved to {output_dir / 'historical_95th_percentile.png'}")
    
    # =============================================================================
    # CALCULATE ANOMALIES
    # =============================================================================
    print("Calculating anomalies...")
    
    ssp370_fwi_95th_annual_count_anomaly = ssp370_fwi_95th_annual_count - historical_fwi_95th_annual_count
    global_fwi_95th_annual_count_anomaly = global_fwi_95th_annual_count - historical_fwi_95th_annual_count
    ssp126_fwi_95th_annual_count_anomaly = ssp126_fwi_95th_annual_count - historical_fwi_95th_annual_count
    aer126eff_fwi_95th_annual_count_anomaly = global_fwi_95th_annual_count_anomaly - ssp370_fwi_95th_annual_count_anomaly
    ghg126eff_fwi_95th_annual_count_anomaly = ssp126_fwi_95th_annual_count_anomaly - global_fwi_95th_annual_count_anomaly
    
    eas_fwi_95th_annual_count_anomaly = eas_fwi_95th_annual_count - ssp370_fwi_95th_annual_count
    sas_fwi_95th_annual_count_anomaly = sas_fwi_95th_annual_count - ssp370_fwi_95th_annual_count
    afr_fwi_95th_annual_count_anomaly = afr_fwi_95th_annual_count - ssp370_fwi_95th_annual_count
    nae_fwi_95th_annual_count_anomaly = nae_fwi_95th_annual_count - ssp370_fwi_95th_annual_count
    
    # =============================================================================
    # APPLY MASKS FOR ANOMALY ANALYSIS
    # =============================================================================
    print("Applying masks for anomaly analysis...")
    
    # Apply masks for global average value (textbox on maps)
    ssp370_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(ssp370_fwi_95th_annual_count_anomaly, 
                                                                        get_significance=False,
                                                                        get_land_mask=True)
    global_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(global_fwi_95th_annual_count_anomaly, 
                                                                        get_significance=False,
                                                                        get_land_mask=True)
    ssp126_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(ssp126_fwi_95th_annual_count_anomaly, 
                                                                        get_significance=False,
                                                                        get_land_mask=True)
    aer126eff_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(aer126eff_fwi_95th_annual_count_anomaly, 
                                                                           get_significance=False,
                                                                           get_land_mask=True)
    ghg126eff_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(ghg126eff_fwi_95th_annual_count_anomaly, 
                                                                           get_significance=False,
                                                                           get_land_mask=True)
    eas_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(eas_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=False,
                                                                     get_land_mask=True)
    sas_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(sas_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=False,
                                                                     get_land_mask=True)
    afr_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(afr_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=False,
                                                                     get_land_mask=True)
    nae_fwi_95th_annual_count_anomaly_masked_textbox, _ = apply_masks(nae_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=False,
                                                                     get_land_mask=True)
    
    # Get latitudinally weighted global average
    ssp370_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        ssp370_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    global_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        global_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    ssp126_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        ssp126_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    aer126eff_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        aer126eff_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    ghg126eff_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        ghg126eff_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    eas_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        eas_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    sas_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        sas_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    afr_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        afr_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    nae_fwi_95th_annual_count_anomaly_masked_globalavg = weighted_horizontal_avg(
        nae_fwi_95th_annual_count_anomaly_masked_textbox.mean('model'), 
        ensemble=False, 
        time=False
    )
    
    # Apply masks for filled contour on maps
    ssp370_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(ssp370_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=True,
                                                                     agreement_threshold=0.66,
                                                                     threshold_type='minimum',
                                                                     get_land_mask=True,
                                                                     baseline_data=ssp370_fwi_95th_annual_count_anomaly)
    global_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(global_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=True,
                                                                     agreement_threshold=0.66,
                                                                     threshold_type='minimum',
                                                                     get_land_mask=True,
                                                                     baseline_data=global_fwi_95th_annual_count_anomaly)
    ssp126_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(ssp126_fwi_95th_annual_count_anomaly, 
                                                                     get_significance=True,
                                                                     agreement_threshold=0.66,
                                                                     threshold_type='minimum',
                                                                     get_land_mask=True,
                                                                     baseline_data=ssp126_fwi_95th_annual_count_anomaly)
    aer126eff_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(aer126eff_fwi_95th_annual_count_anomaly, 
                                                                        get_significance=True,
                                                                        agreement_threshold=0.66,
                                                                        threshold_type='minimum',
                                                                        get_land_mask=True,
                                                                        baseline_data=aer126eff_fwi_95th_annual_count_anomaly)
    ghg126eff_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(ghg126eff_fwi_95th_annual_count_anomaly, 
                                                                        get_significance=True,
                                                                        agreement_threshold=0.66,
                                                                        threshold_type='minimum',
                                                                        get_land_mask=True,
                                                                        baseline_data=ghg126eff_fwi_95th_annual_count_anomaly)
    eas_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(eas_fwi_95th_annual_count_anomaly, 
                                                                  get_significance=True,
                                                                  agreement_threshold=0.66,
                                                                  threshold_type='minimum',
                                                                  get_land_mask=True,
                                                                  baseline_data=eas_fwi_95th_annual_count_anomaly)
    sas_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(sas_fwi_95th_annual_count_anomaly, 
                                                                  get_significance=True,
                                                                  agreement_threshold=0.66,
                                                                  threshold_type='minimum',
                                                                  get_land_mask=True,
                                                                  baseline_data=sas_fwi_95th_annual_count_anomaly)
    afr_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(afr_fwi_95th_annual_count_anomaly, 
                                                                  get_significance=True,
                                                                  agreement_threshold=0.66,
                                                                  threshold_type='minimum',
                                                                  get_land_mask=True,
                                                                  baseline_data=afr_fwi_95th_annual_count_anomaly)
    nae_fwi_95th_annual_count_anomaly_masked_plot, _ = apply_masks(nae_fwi_95th_annual_count_anomaly, 
                                                                  get_significance=True,
                                                                  agreement_threshold=0.66,
                                                                  threshold_type='minimum',
                                                                  get_land_mask=True,
                                                                  baseline_data=nae_fwi_95th_annual_count_anomaly)
    
    # Apply masks for hatching on maps
    ssp370_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(ssp370_fwi_95th_annual_count_anomaly, 
                                                                         get_significance=True,
                                                                         agreement_threshold=0.67,
                                                                         threshold_type='maximum',
                                                                         get_land_mask=True,
                                                                         baseline_data=ssp370_fwi_95th_annual_count_anomaly)
    global_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(global_fwi_95th_annual_count_anomaly, 
                                                                         get_significance=True,
                                                                         agreement_threshold=0.67,
                                                                         threshold_type='maximum',
                                                                         get_land_mask=True,
                                                                         baseline_data=global_fwi_95th_annual_count_anomaly)
    ssp126_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(ssp126_fwi_95th_annual_count_anomaly, 
                                                                         get_significance=True,
                                                                         agreement_threshold=0.67,
                                                                         threshold_type='maximum',
                                                                         get_land_mask=True,
                                                                         baseline_data=ssp126_fwi_95th_annual_count_anomaly)
    aer126eff_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(aer126eff_fwi_95th_annual_count_anomaly, 
                                                                            get_significance=True,
                                                                            agreement_threshold=0.67,
                                                                            threshold_type='maximum',
                                                                            get_land_mask=True,
                                                                            baseline_data=aer126eff_fwi_95th_annual_count_anomaly)
    ghg126eff_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(ghg126eff_fwi_95th_annual_count_anomaly, 
                                                                           get_significance=True,
                                                                           agreement_threshold=0.67,
                                                                           threshold_type='maximum',
                                                                           get_land_mask=True,
                                                                           baseline_data=ghg126eff_fwi_95th_annual_count_anomaly)
    eas_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(eas_fwi_95th_annual_count_anomaly, 
                                                                      get_significance=True,
                                                                      agreement_threshold=0.67,
                                                                      threshold_type='maximum',
                                                                      get_land_mask=True,
                                                                      baseline_data=eas_fwi_95th_annual_count_anomaly)
    sas_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(sas_fwi_95th_annual_count_anomaly, 
                                                                      get_significance=True,
                                                                      agreement_threshold=0.67,
                                                                      threshold_type='maximum',
                                                                      get_land_mask=True,
                                                                      baseline_data=sas_fwi_95th_annual_count_anomaly)
    afr_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(afr_fwi_95th_annual_count_anomaly, 
                                                                      get_significance=True,
                                                                      agreement_threshold=0.67,
                                                                      threshold_type='maximum',
                                                                      get_land_mask=True,
                                                                      baseline_data=afr_fwi_95th_annual_count_anomaly)
    nae_fwi_95th_annual_count_anomaly_masked_hatching, _ = apply_masks(nae_fwi_95th_annual_count_anomaly, 
                                                                      get_significance=True,
                                                                      agreement_threshold=0.67,
                                                                      threshold_type='maximum',
                                                                      get_land_mask=True,
                                                                      baseline_data=nae_fwi_95th_annual_count_anomaly)
    
    # =============================================================================
    # CREATE ANOMALY PLOTS
    # =============================================================================
    print("Creating anomaly plots...")
    
    # Main scenarios grid
    data_list = [
        ssp370_fwi_95th_annual_count_anomaly_masked_plot.mean('model'), 
        global_fwi_95th_annual_count_anomaly_masked_plot.mean('model'), 
        ssp126_fwi_95th_annual_count_anomaly_masked_plot.mean('model'),
        aer126eff_fwi_95th_annual_count_anomaly_masked_plot.mean('model'),
        ghg126eff_fwi_95th_annual_count_anomaly_masked_plot.mean('model')
    ]
    
    textbox_text_list = [
        f"{ssp370_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{global_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{ssp126_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{aer126eff_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{ghg126eff_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days"
    ]
    
    hatching_data_list = [
        ssp370_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        global_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        ssp126_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        aer126eff_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        ghg126eff_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0)
    ]
    
    fig, axes = create_global_map_grid(
        data_list=data_list,
        rows=2, cols=3,
        main_title="95th Percentile Exceedances Changes (2041-2050 vs 1961-1990) - Main Scenarios",
        titles=['SSP3-7.0', 'Global Aerosol Reduction', 'SSP1-2.6', 
                'Aerosol Effect', 'GHG Effect'],
        colormaps='RdBu_r',
        colorbar_title="Δ Days per Year",
        textbox_texts=textbox_text_list,
        hatching_data=hatching_data_list,
        figsize=(20, 12)
    )
    
    # Save main scenarios plot
    plt.savefig(output_dir / "95th_percentile_anomalies_main_scenarios.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Main scenarios plot saved to {output_dir / '95th_percentile_anomalies_main_scenarios.png'}")
    
    # Regional scenarios grid
    regional_data_list = [
        eas_fwi_95th_annual_count_anomaly_masked_plot.mean('model'),
        sas_fwi_95th_annual_count_anomaly_masked_plot.mean('model'),
        afr_fwi_95th_annual_count_anomaly_masked_plot.mean('model'),
        nae_fwi_95th_annual_count_anomaly_masked_plot.mean('model')
    ]
    
    regional_textbox_text_list = [
        f"{eas_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{sas_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{afr_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days",
        f"{nae_fwi_95th_annual_count_anomaly_masked_globalavg.values.item():.1f} days"
    ]
    
    regional_hatching_data_list = [
        eas_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        sas_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        afr_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0),
        nae_fwi_95th_annual_count_anomaly_masked_hatching.isel(model=0)
    ]
    
    fig, axes = create_global_map_grid(
        data_list=regional_data_list,
        rows=2, cols=2,
        main_title="95th Percentile Exceedances Changes (2041-2050 vs 1961-1990) - Regional Aerosol Reductions",
        titles=['East Asia', 'South Asia', 'Africa & Middle East', 'North America & Europe'],
        colormaps='RdBu_r',
        colorbar_title="Δ Days per Year",
        textbox_texts=regional_textbox_text_list,
        hatching_data=regional_hatching_data_list,
        figsize=(16, 12)
    )
    
    # Save regional scenarios plot
    plt.savefig(output_dir / "95th_percentile_anomalies_regional_scenarios.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regional scenarios plot saved to {output_dir / '95th_percentile_anomalies_regional_scenarios.png'}")
    
    print("Annual 95th Percentile Exceedances Analysis complete!")
    print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
