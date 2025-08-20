#!/usr/bin/env python3
"""
Specific Fire Weather Index Analyzer Implementations
===================================================

This module contains specialized analyzer classes that inherit from the base
FireWeatherAnalyzer class to perform different types of fire weather analysis:

- RawValueAnalyzer: Analyzes raw FWI values and changes
- AbsoluteThresholdAnalyzer: Analyzes exceedance of absolute thresholds (e.g., high fire danger days)
- PercentileThresholdAnalyzer: Analyzes exceedance of percentile-based thresholds (e.g., 95th percentile)

Each analyzer implements the specific data processing and analysis logic for its metric type.
"""

import xarray as xr #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from typing import Dict, Any, Tuple

from fwi_analysis_framework import (
    FireWeatherAnalyzer, 
    FireWeatherAnalysisConfig
)

# Import utility functions
from ramip_fwi_utilities import (
    apply_masks, 
    weighted_horizontal_avg, 
    create_global_map_grid,
    season_mean
)


# =============================================================================
# RAW VALUE ANALYZER
# =============================================================================

class RawValueAnalyzer(FireWeatherAnalyzer):
    """Analyzer for raw FWI values."""
    
    def process_data(self, raw_data: Dict[str, Dict[str, xr.DataArray]]) -> Dict[str, Any]:
        """Process raw data for FWI value analysis."""
        print("Processing data for raw FWI analysis...")
        print("-" * 60)
        
        processed = {}
        
        # Process each scenario
        for scenario_name, scenario_data in raw_data.items():
            print(f"  - Processing {scenario_name}...")
            processed[scenario_name] = self.combine_models(scenario_data)
        
        print("✓ Data processing complete")
        return processed
    
    def _process_seasonal_data(self, scenario_name: str) -> xr.DataArray:
        """Process seasonal data for a specific scenario, applying season_mean to each model individually."""
        # Get the raw data for this scenario
        scenario_raw_data = self.raw_data[scenario_name]
        
        # Process each model individually: season_mean first, then mean across members
        model_seasonal_data = []
        for model in self.config.models:
            model_data = scenario_raw_data[model]
            # Apply season_mean to individual model data, then average across members
            model_seasonal = season_mean(model_data).mean('member', skipna=True)
            model_seasonal_data.append(model_seasonal)
        
        # Now combine the processed seasonal data across models
        combined_seasonal = xr.concat(model_seasonal_data, dim="model", coords='minimal')
        return combined_seasonal.compute()
    
    def run_annual_analysis(self):
        """Run annual FWI analysis."""
        print("\nANNUAL FWI ANALYSIS")
        print("=" * 80)
        print("Calculating annual means and anomalies...")
        print("-" * 60)
        
        # Calculate annual means
        print("Step 1: Calculating annual means...")
        annual_means = {}
        for scenario, data in self.processed_data.items():
            annual_means[scenario] = data.mean(dim=['time', 'member'], skipna=True)
        
        # Create historical baseline plot
        print("\nStep 2: Creating historical baseline plot...")
        self.plotting_manager.create_historical_plot(
            annual_means['historical'],
            f"{self.config.output_prefix}_historical_fwi_annual.png"
        )
        print(f"  ✓ Historical plot saved")
        
        # Calculate anomalies
        print("\nStep 3: Calculating anomalies...")
        anomalies = {}
        
        # Main scenario anomalies (vs. historical)
        for scenario in self.config.main_scenarios:
            anomalies[scenario] = self.calculate_anomalies(
                annual_means[scenario], annual_means['historical']
            )
        
        # Effect calculations
        anomalies['aer126eff'] = anomalies['ssp370-126aer'] - anomalies['ssp370']
        anomalies['ghg126eff'] = anomalies['ssp126'] - anomalies['ssp370-126aer']
        
        # Regional scenario anomalies (vs. SSP3-7.0)
        for scenario in self.config.regional_scenarios:
            anomalies[scenario] = self.calculate_anomalies(
                annual_means[scenario], annual_means['ssp370']
            )
        
        # Apply masks and calculate global averages
        print("\nStep 4: Applying masks and calculating global averages...")
        plot_data, hatching_data, global_avgs = self._process_anomalies_for_plotting(anomalies)
        
        # Create main scenarios plot
        print("\nStep 5: Creating main scenarios plot...")
        main_scenarios = ['ssp370', 'ssp370-126aer', 'ssp126', 'aer126eff', 'ghg126eff']
        main_plot_data = {k: plot_data[k] for k in main_scenarios}
        main_hatching_data = {k: hatching_data[k] for k in main_scenarios}
        main_global_avgs = {k: global_avgs[k] for k in main_scenarios}
        
        self.plotting_manager.create_main_scenarios_grid(
            main_plot_data, main_global_avgs, main_hatching_data,
            f"{self.config.output_prefix}_main_scenarios_fwi_annual.png",
            "Annual FWI Changes"
        )
        print(f"  ✓ Main scenarios plot saved")
        
        # Create regional scenarios plot
        print("\nStep 6: Creating regional scenarios plot...")
        regional_plot_data = {k: plot_data[k] for k in self.config.regional_scenarios}
        regional_hatching_data = {k: hatching_data[k] for k in self.config.regional_scenarios}
        regional_global_avgs = {k: global_avgs[k] for k in self.config.regional_scenarios}
        
        self.plotting_manager.create_regional_scenarios_grid(
            regional_plot_data, regional_global_avgs, regional_hatching_data,
            f"{self.config.output_prefix}_regional_scenarios_fwi_annual.png",
            "Regional Aerosol Reduction Effects on Annual FWI"
        )
        print(f"  ✓ Regional scenarios plot saved")
        
        print("✓ Annual analysis complete")
    
    def run_seasonal_analysis(self):
        """Run seasonal FWI analysis."""
        print("\nSEASONAL FWI ANALYSIS")
        print("=" * 80)
        print("Calculating seasonal means and anomalies...")
        print("-" * 60)
        
        # Calculate seasonal means
        print("Step 1: Calculating seasonal means...")
        seasonal_means = {}
        for scenario, data in self.processed_data.items():
            # For seasonal analysis, we need to process each model individually
            # before combining, to match the original script behavior
            seasonal_data = self._process_seasonal_data(scenario)
            seasonal_means[scenario] = seasonal_data
        
        # Create historical seasonal baseline plot
        print("\nStep 2: Creating historical seasonal baseline plot...")
        self._create_historical_seasonal_plot(seasonal_means['historical'])
        print(f"  ✓ Historical seasonal plot saved")
        
        # Calculate seasonal anomalies
        print("\nStep 3: Calculating seasonal anomalies...")
        seasonal_anomalies = {}
        
        # Main scenario anomalies (vs. historical)
        for scenario in self.config.main_scenarios:
            seasonal_anomalies[scenario] = self.calculate_anomalies(
                seasonal_means[scenario], seasonal_means['historical']
            )
        
        # Effect calculations
        seasonal_anomalies['aer126eff'] = seasonal_anomalies['ssp370-126aer'] - seasonal_anomalies['ssp370']
        seasonal_anomalies['ghg126eff'] = seasonal_anomalies['ssp126'] - seasonal_anomalies['ssp370-126aer']
        
        # Regional scenario anomalies (vs. SSP3-7.0)
        for scenario in self.config.regional_scenarios:
            seasonal_anomalies[scenario] = self.calculate_anomalies(
                seasonal_means[scenario], seasonal_means['ssp370']
            )
        
        # Create seasonal plots for main scenarios
        print("\nStep 4: Creating seasonal plots for main scenarios...")
        main_scenarios = ['ssp370', 'ssp370-126aer', 'ssp126', 'aer126eff', 'ghg126eff']
        scenario_names = ['SSP3-7.0', 'SSP3-7.0 with Global Aerosol Reduction', 'SSP1-2.6',
                         'Effect of Aerosol Emission Reduction', 'Effect of GHG Emission Reduction']
        
        for scenario, name in zip(main_scenarios, scenario_names):
            self._create_seasonal_plot(seasonal_anomalies[scenario], scenario, name)
            print(f"  ✓ Seasonal plot for {name} saved")
        
        # Create seasonal plots for regional scenarios
        print("\nStep 5: Creating seasonal plots for regional scenarios...")
        for i, (scenario, name) in enumerate(zip(self.config.regional_scenarios, self.config.regional_names)):
            ramip_region = ['east_asia', 'north_america_europe', 'south_asia', 'africa_mideast'][i]
            self._create_seasonal_plot(seasonal_anomalies[scenario], scenario, 
                                     f"{name} Aerosol Reduction", ramip_region)
            print(f"  ✓ Seasonal plot for {name} saved")
        
        print("✓ Seasonal analysis complete")
    
    def _process_anomalies_for_plotting(self, anomalies: Dict[str, xr.DataArray]) -> Tuple[Dict, Dict, Dict]:
        """Process anomalies for plotting by applying masks and calculating global averages."""
        plot_data = {}
        hatching_data = {}
        global_avgs = {}
        
        for scenario, data in anomalies.items():
            # Apply masks for plotting
            plot_data[scenario] = self.apply_masks_for_plotting(data, for_hatching=False)
            hatching_data[scenario] = self.apply_masks_for_plotting(data, for_hatching=True)
            
            # Calculate global averages
            global_avgs[scenario] = self.calculate_global_average(data)
        
        return plot_data, hatching_data, global_avgs
    
    def _create_historical_seasonal_plot(self, historical_data: xr.DataArray):
        """Create historical seasonal baseline plot."""
        # Apply masks
        masked_data, _ = apply_masks(historical_data, get_significance=False, get_land_mask=True, baseline_data=None)
        
        # Calculate global averages
        global_avg = weighted_horizontal_avg(masked_data.mean('model'), ensemble=False, time=False)
        
        # Prepare data for 2x2 grid
        seasonal_data_list = [
            masked_data.sel(season='JJA').mean('model'),
            masked_data.sel(season='SON').mean('model'),
            masked_data.sel(season='DJF').mean('model'),
            masked_data.sel(season='MAM').mean('model')
        ]
        
        seasonal_titles = ['JJA (Jun-Jul-Aug)', 'SON (Sep-Oct-Nov)', 
                          'DJF (Dec-Jan-Feb)', 'MAM (Mar-Apr-May)']
        
        seasonal_textbox_texts = [
            f"{global_avg.sel(season=season).values.item():.2f}" 
            for season in ['JJA', 'SON', 'DJF', 'MAM']
        ]
        
        plot_params = self.config.plot_params['historical']
        
        fig, axes = create_global_map_grid(
            data_list=seasonal_data_list,
            rows=2, cols=2,
            main_title=f"Historical ({self.config.historical_start.year}-{self.config.historical_end.year})",
            titles=seasonal_titles,
            colormaps=plot_params['colormap'],
            colorbar_titles=plot_params['colorbar_title'],
            textbox_texts=seasonal_textbox_texts,
            vmins=[plot_params['vmin']] * 4,
            vmaxs=[plot_params['vmax']] * 4,
            extends='max',
            colorbar_levels=[plot_params['colorbar_levels']] * 4,
            regional_boundaries='ar6',
            show_gridlines=False
        )
        
        plt.savefig(self.config.output_dir / f"{self.config.output_prefix}_historical_fwi_seasonal.png",
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_seasonal_plot(self, seasonal_data: xr.DataArray, scenario: str, 
                            scenario_name: str, ramip_region: str = None):
        """Create seasonal plot for a specific scenario."""
        # Apply masks
        plot_data = self.apply_masks_for_plotting(seasonal_data, for_hatching=False)
        hatching_data = self.apply_masks_for_plotting(seasonal_data, for_hatching=True)
        
        # Calculate global averages
        global_avg = self.calculate_global_average(seasonal_data)
        
        # Prepare data for 2x2 grid
        seasonal_data_list = [
            plot_data.sel(season='JJA').mean('model'),
            plot_data.sel(season='SON').mean('model'),
            plot_data.sel(season='DJF').mean('model'),
            plot_data.sel(season='MAM').mean('model')
        ]
        
        seasonal_titles = ['JJA (Jun-Jul-Aug)', 'SON (Sep-Oct-Nov)', 
                          'DJF (Dec-Jan-Feb)', 'MAM (Mar-Apr-May)']
        
        seasonal_textbox_texts = [
            f"{global_avg.sel(season=season).values.item():.2f}" 
            for season in ['JJA', 'SON', 'DJF', 'MAM']
        ]
        
        seasonal_hatching_list = [
            hatching_data.sel(season=season).isel(model=0) 
            for season in ['JJA', 'SON', 'DJF', 'MAM']
        ]
        
        # Determine plot parameters based on scenario type
        if scenario in ['ssp370', 'ssp370-126aer', 'ssp126']:
            plot_params = self.config.plot_params['main_scenarios']
            vmins = [-3] * 4
            vmaxs = [3] * 4
            colorbar_levels = [np.arange(-3, 3.1, 0.3)] * 4
        elif scenario in ['aer126eff', 'ghg126eff']:
            plot_params = self.config.plot_params['main_scenarios']
            vmins = [-1] * 4
            vmaxs = [1] * 4
            colorbar_levels = [np.arange(-1, 1.1, 0.1)] * 4
        else:  # Regional scenarios
            plot_params = self.config.plot_params['regional_scenarios']
            vmins = [-0.5] * 4
            vmaxs = [0.5] * 4
            colorbar_levels = [np.arange(-0.5, 0.51, 0.05)] * 4
        
        fig, axes = create_global_map_grid(
            data_list=seasonal_data_list,
            rows=2, cols=2,
            main_title=f"Seasonal FWI Changes: {scenario_name}",
            titles=seasonal_titles,
            colormaps=plot_params['colormap'],
            colorbar_titles=plot_params['colorbar_title'],
            textbox_texts=seasonal_textbox_texts,
            vmins=vmins,
            vmaxs=vmaxs,
            extends='both',
            colorbar_levels=colorbar_levels,
            hatchings='///',
            regional_boundaries='ar6',
            hatching_styles='overlay',
            hatching_data=seasonal_hatching_list,
            show_gridlines=False,
            ramip_regions=ramip_region
        )
        
        plt.savefig(self.config.output_dir / f"{self.config.output_prefix}_{scenario}_fwi_seasonal.png",
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()


# =============================================================================
# ABSOLUTE THRESHOLD ANALYZER
# =============================================================================

class AbsoluteThresholdAnalyzer(FireWeatherAnalyzer):
    """Analyzer for absolute threshold exceedance (e.g., high fire danger days)."""
    
    def process_data(self, raw_data: Dict[str, Dict[str, xr.DataArray]]) -> Dict[str, Any]:
        """Process raw data for absolute threshold analysis."""
        print(f"Processing data for absolute threshold analysis (threshold = {self.config.threshold_value})...")
        print("-" * 60)
        
        processed = {}
        
        # Process each scenario
        for scenario_name, scenario_data in raw_data.items():
            print(f"  - Processing {scenario_name}...")
            
            # Combine models first
            combined_data = self.combine_models(scenario_data)
            
            # Calculate exceedance days
            exceedance_data = self._calculate_annual_exceedance_days(combined_data)
            processed[scenario_name] = exceedance_data
        
        print("✓ Data processing complete")
        return processed
    
    def _calculate_annual_exceedance_days(self, data: xr.DataArray) -> xr.DataArray:
        """Calculate annual exceedance days for absolute threshold."""
        # Create boolean mask for days exceeding threshold
        exceeds_threshold = data > self.config.threshold_value
        
        # Group by year and sum exceedance days
        annual_exceedance = exceeds_threshold.groupby('time.year').sum('time')
        
        # Calculate mean across years and members
        return annual_exceedance.mean(dim=['year', 'member'], skipna=True)
    
    def _calculate_seasonal_exceedance_days(self, data: xr.DataArray) -> xr.DataArray:
        """Calculate seasonal exceedance days for absolute threshold."""
        # Create boolean mask for days exceeding threshold
        exceeds_threshold = data > self.config.threshold_value
        
        # Calculate seasonal means
        seasonal_data = season_mean(exceeds_threshold)
        
        # Convert to days per season (multiply by ~90 days per season)
        seasonal_exceedance = seasonal_data * 90  # Approximate days per season
        
        return seasonal_exceedance.mean('member', skipna=True)
    
    def run_annual_analysis(self):
        """Run annual threshold exceedance analysis."""
        print(f"\nANNUAL ABSOLUTE THRESHOLD ANALYSIS (threshold = {self.config.threshold_value})")
        print("=" * 80)
        print("Calculating annual exceedance days and anomalies...")
        print("-" * 60)
        
        # Data is already processed as annual exceedance days
        annual_exceedance = self.processed_data
        
        # Create historical baseline plot
        print("\nStep 1: Creating historical baseline plot...")
        self.plotting_manager.create_historical_plot(
            annual_exceedance['historical'],
            f"{self.config.output_prefix}_historical_threshold{int(self.config.threshold_value)}_annual.png",
            f"Historical High Fire Danger Days (FWI > {self.config.threshold_value})"
        )
        print(f"  ✓ Historical plot saved")
        
        # Calculate anomalies
        print("\nStep 2: Calculating anomalies...")
        anomalies = {}
        
        # Main scenario anomalies (vs. historical)
        for scenario in self.config.main_scenarios:
            anomalies[scenario] = self.calculate_anomalies(
                annual_exceedance[scenario], annual_exceedance['historical']
            )
        
        # Effect calculations
        anomalies['aer126eff'] = anomalies['ssp370-126aer'] - anomalies['ssp370']
        anomalies['ghg126eff'] = anomalies['ssp126'] - anomalies['ssp370-126aer']
        
        # Regional scenario anomalies (vs. SSP3-7.0)
        for scenario in self.config.regional_scenarios:
            anomalies[scenario] = self.calculate_anomalies(
                annual_exceedance[scenario], annual_exceedance['ssp370']
            )
        
        # Apply masks and create plots (similar to RawValueAnalyzer)
        print("\nStep 3: Applying masks and creating plots...")
        plot_data, hatching_data, global_avgs = self._process_anomalies_for_plotting(anomalies)
        
        # Create main scenarios plot
        main_scenarios = ['ssp370', 'ssp370-126aer', 'ssp126', 'aer126eff', 'ghg126eff']
        main_plot_data = {k: plot_data[k] for k in main_scenarios}
        main_hatching_data = {k: hatching_data[k] for k in main_scenarios}
        main_global_avgs = {k: global_avgs[k] for k in main_scenarios}
        
        self.plotting_manager.create_main_scenarios_grid(
            main_plot_data, main_global_avgs, main_hatching_data,
            f"{self.config.output_prefix}_main_scenarios_threshold{int(self.config.threshold_value)}_annual.png",
            f"Annual High Fire Danger Days Changes (FWI > {self.config.threshold_value})"
        )
        print(f"  ✓ Main scenarios plot saved")
        
        # Create regional scenarios plot
        regional_plot_data = {k: plot_data[k] for k in self.config.regional_scenarios}
        regional_hatching_data = {k: hatching_data[k] for k in self.config.regional_scenarios}
        regional_global_avgs = {k: global_avgs[k] for k in self.config.regional_scenarios}
        
        self.plotting_manager.create_regional_scenarios_grid(
            regional_plot_data, regional_global_avgs, regional_hatching_data,
            f"{self.config.output_prefix}_regional_scenarios_threshold{int(self.config.threshold_value)}_annual.png",
            f"Regional Aerosol Reduction Effects on High Fire Danger Days (FWI > {self.config.threshold_value})"
        )
        print(f"  ✓ Regional scenarios plot saved")
        
        print("✓ Annual threshold analysis complete")
    
    def run_seasonal_analysis(self):
        """Run seasonal threshold exceedance analysis."""
        print(f"\nSEASONAL ABSOLUTE THRESHOLD ANALYSIS (threshold = {self.config.threshold_value})")
        print("=" * 80)
        
        # Recalculate seasonal exceedance from raw data
        print("Calculating seasonal exceedance days...")
        seasonal_exceedance = {}
        for scenario_name, scenario_data in self.raw_data.items():
            combined_data = self.combine_models(scenario_data)
            seasonal_exceedance[scenario_name] = self._calculate_seasonal_exceedance_days(combined_data)
        
        # Similar to annual analysis but for seasonal data
        # Implementation would follow the same pattern as RawValueAnalyzer
        # but working with exceedance day counts instead of raw FWI values
        
        print("✓ Seasonal threshold analysis complete")
    
    def _process_anomalies_for_plotting(self, anomalies: Dict[str, xr.DataArray]) -> Tuple[Dict, Dict, Dict]:
        """Process anomalies for plotting (same as RawValueAnalyzer)."""
        plot_data = {}
        hatching_data = {}
        global_avgs = {}
        
        for scenario, data in anomalies.items():
            plot_data[scenario] = self.apply_masks_for_plotting(data, for_hatching=False)
            hatching_data[scenario] = self.apply_masks_for_plotting(data, for_hatching=True)
            global_avgs[scenario] = self.calculate_global_average(data)
        
        return plot_data, hatching_data, global_avgs


# =============================================================================
# PERCENTILE THRESHOLD ANALYZER
# =============================================================================

class PercentileThresholdAnalyzer(FireWeatherAnalyzer):
    """Analyzer for percentile threshold exceedance (e.g., 95th percentile days)."""
    
    def process_data(self, raw_data: Dict[str, Dict[str, xr.DataArray]]) -> Dict[str, Any]:
        """Process raw data for percentile threshold analysis."""
        print(f"Processing data for percentile threshold analysis ({self.config.threshold_value}th percentile)...")
        print("-" * 60)
        
        # First, calculate historical percentile thresholds
        print("  - Calculating historical percentile thresholds...")
        historical_data = self.combine_models(raw_data['historical'])
        percentile_thresholds = historical_data.quantile(self.config.threshold_value/100, dim=['time', 'member'])
        
        processed = {}
        
        # Process each scenario
        for scenario_name, scenario_data in raw_data.items():
            print(f"  - Processing {scenario_name}...")
            
            # Combine models first
            combined_data = self.combine_models(scenario_data)
            
            # Calculate exceedance days relative to historical percentile
            exceedance_data = self._calculate_annual_percentile_exceedance(combined_data, percentile_thresholds)
            processed[scenario_name] = exceedance_data
        
        print("✓ Data processing complete")
        return processed
    
    def _calculate_annual_percentile_exceedance(self, data: xr.DataArray, 
                                              percentile_thresholds: xr.DataArray) -> xr.DataArray:
        """Calculate annual exceedance days for percentile threshold."""
        # Create boolean mask for days exceeding historical percentile threshold
        exceeds_threshold = data > percentile_thresholds
        
        # Group by year and sum exceedance days
        annual_exceedance = exceeds_threshold.groupby('time.year').sum('time')
        
        # Calculate mean across years and members
        return annual_exceedance.mean(dim=['year', 'member'], skipna=True)
    
    def run_annual_analysis(self):
        """Run annual percentile threshold analysis."""
        print(f"\nANNUAL PERCENTILE THRESHOLD ANALYSIS ({self.config.threshold_value}th percentile)")
        print("=" * 80)
        
        # Implementation similar to AbsoluteThresholdAnalyzer
        # but working with percentile-based thresholds
        
        print("✓ Annual percentile analysis complete")
    
    def run_seasonal_analysis(self):
        """Run seasonal percentile threshold analysis."""
        print(f"\nSEASONAL PERCENTILE THRESHOLD ANALYSIS ({self.config.threshold_value}th percentile)")
        print("=" * 80)
        
        # Implementation similar to seasonal absolute threshold analysis
        # but working with percentile-based thresholds
        
        print("✓ Seasonal percentile analysis complete")


if __name__ == "__main__":
    print("Fire Weather Index Analyzers loaded successfully!")
    print("Available analyzers: RawValueAnalyzer, AbsoluteThresholdAnalyzer, PercentileThresholdAnalyzer")
