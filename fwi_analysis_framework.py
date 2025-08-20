#!/usr/bin/env python3
"""
Modular Fire Weather Index Analysis Framework
===========================================

This module provides a flexible, modular framework for analyzing fire weather indices
across different metrics (raw values, absolute thresholds, percentile thresholds) and
temporal scales (annual, seasonal).

Key Components:
- FireWeatherAnalysisConfig: Configuration management for different analysis types
- DataManager: Handles data loading, preprocessing, and regridding
- FireWeatherAnalyzer: Base class for analysis with common functionality
- PlottingManager: Handles visualization and map generation
- Specialized analyzers: RawValueAnalyzer, AbsoluteThresholdAnalyzer, PercentileThresholdAnalyzer

Example Usage:
-------------
# Raw FWI values analysis
config = FireWeatherAnalysisConfig.for_raw_values()
analyzer = RawValueAnalyzer(config)
analyzer.run_analysis()

# Absolute threshold analysis (high fire danger days)
config = FireWeatherAnalysisConfig.for_absolute_threshold(threshold=20)
analyzer = AbsoluteThresholdAnalyzer(config)
analyzer.run_analysis()

# Percentile threshold analysis
config = FireWeatherAnalysisConfig.for_percentile_threshold(percentile=95)
analyzer = PercentileThresholdAnalyzer(config)
analyzer.run_analysis()
"""

import xarray as xr #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import cartopy.crs as ccrs #type: ignore
import cftime #type: ignore
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import utility functions
from ramip_fwi_utilities import (
    read_zarr, 
    apply_masks, 
    weighted_horizontal_avg, 
    create_global_map, 
    create_global_map_grid,
    season_mean
)


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class FireWeatherAnalysisConfig:
    """Configuration class for fire weather analysis parameters."""
    
    # Analysis type and parameters
    analysis_type: str  # 'raw_values', 'absolute_threshold', 'percentile_threshold'
    variable_name: str = 'FWI'
    threshold_value: Optional[float] = None
    
    # Time periods
    historical_start: cftime.DatetimeNoLeap = field(default_factory=lambda: cftime.DatetimeNoLeap(1961, 1, 1, 12, 0, 0, 0, has_year_zero=True))
    historical_end: cftime.DatetimeNoLeap = field(default_factory=lambda: cftime.DatetimeNoLeap(1990, 12, 31, 12, 0, 0, 0, has_year_zero=True))
    future_start: cftime.DatetimeNoLeap = field(default_factory=lambda: cftime.DatetimeNoLeap(2041, 1, 1, 12, 0, 0, 0, has_year_zero=True))
    future_end: cftime.DatetimeNoLeap = field(default_factory=lambda: cftime.DatetimeNoLeap(2050, 12, 31, 12, 0, 0, 0, has_year_zero=True))
    
    # Models and scenarios
    models: List[str] = field(default_factory=lambda: ['NorESM2-LM', 'SPEAR', 'MRI-ESM2-0'])
    main_scenarios: List[str] = field(default_factory=lambda: ['ssp370', 'ssp370-126aer', 'ssp126'])
    regional_scenarios: List[str] = field(default_factory=lambda: ['eas', 'nae', 'sas', 'afr'])
    regional_names: List[str] = field(default_factory=lambda: [
        'East Asia', 'North America & Europe', 'South Asia', 'Africa & Middle East'
    ])
    
    # Statistical parameters
    agreement_threshold_plot: float = 0.66
    agreement_threshold_hatching: float = 0.67
    
    # Output parameters
    output_dir: Path = field(default_factory=lambda: Path("fwi_analysis_plots"))
    output_prefix: str = "mm"
    dpi: int = 300
    
    # Analysis options
    include_annual: bool = True
    include_seasonal: bool = True
    seasons: List[str] = field(default_factory=lambda: ['JJA', 'SON', 'DJF', 'MAM'])
    
    # Plotting parameters for different analysis types
    plot_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def for_raw_values(cls, **kwargs) -> 'FireWeatherAnalysisConfig':
        """Create configuration for raw FWI values analysis."""
        default_plot_params = {
            'main_scenarios': {
                'vmins': [-3, -3, -3, -1, -1],
                'vmaxs': [3, 3, 3, 1, 1],
                'colorbar_levels': [
                    np.arange(-3, 3.1, 0.3), np.arange(-3, 3.1, 0.3), np.arange(-3, 3.1, 0.3),
                    np.arange(-1, 1.1, 0.1), np.arange(-1, 1.1, 0.1)
                ],
                'colormap': 'RdBu_r',
                'colorbar_title': 'Δ Fire Weather Index'
            },
            'regional_scenarios': {
                'vmins': [-0.5, -0.5, -0.5, -0.5],
                'vmaxs': [0.5, 0.5, 0.5, 0.5],
                'colorbar_levels': [np.arange(-0.5, 0.51, 0.05)] * 4,
                'colormap': 'RdBu_r',
                'colorbar_title': 'Δ Fire Weather Index'
            },
            'historical': {
                'vmin': 0,
                'vmax': 40,
                'colorbar_levels': np.arange(0, 40.1, 4),
                'colormap': 'Reds',
                'colorbar_title': 'Fire Weather Index'
            }
        }
        
        return cls(
            analysis_type='raw_values',
            plot_params=default_plot_params,
            **kwargs
        )
    
    @classmethod
    def for_absolute_threshold(cls, threshold: float = 30, **kwargs) -> 'FireWeatherAnalysisConfig':
        """Create configuration for absolute threshold analysis (e.g., high fire danger days)."""
        default_plot_params = {
            'main_scenarios': {
                'vmins': [-30, -30, -30, -10, -10],
                'vmaxs': [30, 30, 30, 10, 10],
                'colorbar_levels': [
                    np.arange(-30, 31, 3), np.arange(-30, 31, 3), np.arange(-30, 31, 3),
                    np.arange(-10, 11, 1), np.arange(-10, 11, 1)
                ],
                'colormap': 'RdBu_r',
                'colorbar_title': 'Δ High Fire Danger Days'
            },
            'regional_scenarios': {
                'vmins': [-5, -5, -5, -5],
                'vmaxs': [5, 5, 5, 5],
                'colorbar_levels': [np.arange(-5, 5.1, 0.5)] * 4,
                'colormap': 'RdBu_r',
                'colorbar_title': 'Δ High Fire Danger Days'
            },
            'historical': {
                'vmin': 0,
                'vmax': 100,
                'colorbar_levels': np.arange(0, 101, 10),
                'colormap': 'Reds',
                'colorbar_title': 'High Fire Danger Days'
            }
        }
        
        return cls(
            analysis_type='absolute_threshold',
            threshold_value=threshold,
            plot_params=default_plot_params,
            **kwargs
        )
    
    @classmethod
    def for_percentile_threshold(cls, percentile: float = 95, **kwargs) -> 'FireWeatherAnalysisConfig':
        """Create configuration for percentile threshold analysis."""
        default_plot_params = {
            'main_scenarios': {
                'vmins': [-20, -20, -20, -8, -8],
                'vmaxs': [20, 20, 20, 8, 8],
                'colorbar_levels': [
                    np.arange(-20, 21, 2), np.arange(-20, 21, 2), np.arange(-20, 21, 2),
                    np.arange(-8, 9, 0.8), np.arange(-8, 9, 0.8)
                ],
                'colormap': 'RdBu_r',
                'colorbar_title': f'Δ Days Above {percentile}th Percentile'
            },
            'regional_scenarios': {
                'vmins': [-3, -3, -3, -3],
                'vmaxs': [3, 3, 3, 3],
                'colorbar_levels': [np.arange(-3, 3.1, 0.3)] * 4,
                'colormap': 'RdBu_r',
                'colorbar_title': f'Δ Days Above {percentile}th Percentile'
            },
            'historical': {
                'vmin': 0,
                'vmax': 50,
                'colorbar_levels': np.arange(0, 51, 5),
                'colormap': 'Reds',
                'colorbar_title': f'Days Above {percentile}th Percentile'
            }
        }
        
        return cls(
            analysis_type='percentile_threshold',
            threshold_value=percentile,
            plot_params=default_plot_params,
            **kwargs
        )


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class DataManager:
    """Handles data loading, preprocessing, and regridding operations."""
    
    def __init__(self, config: FireWeatherAnalysisConfig):
        self.config = config
        self.reference_grid = None
        
    def load_all_data(self) -> Dict[str, Dict[str, xr.DataArray]]:
        """Load all required data for the analysis."""
        print("Loading data...")
        print("-" * 60)
        
        data = {}
        
        # Load historical data
        print("Loading historical baseline data...")
        data['historical'] = self._load_scenario_data('historical', 
                                                     self.config.historical_start, 
                                                     self.config.historical_end)
        print("✓ Historical data loaded")
        
        # Load main scenarios
        print("\nLoading future scenario data...")
        for scenario in self.config.main_scenarios:
            print(f"  - Loading {scenario}...")
            data[scenario] = self._load_scenario_data(scenario,
                                                     self.config.future_start,
                                                     self.config.future_end)
        
        # Load regional scenarios
        print("  - Loading regional scenarios...")
        for scenario in self.config.regional_scenarios:
            scenario_name = f'ssp370-{scenario}126aer'
            print(f"    {scenario} ({scenario_name})")
            data[scenario] = self._load_scenario_data(scenario_name,
                                                     self.config.future_start,
                                                     self.config.future_end)
        
        print("✓ All data loading complete")
        return data
    
    def _load_scenario_data(self, scenario: str, start_time: cftime.DatetimeNoLeap, 
                           end_time: cftime.DatetimeNoLeap) -> Dict[str, xr.DataArray]:
        """Load data for a specific scenario across all models."""
        scenario_data = {}
        
        for model in self.config.models:
            scenario_data[model] = read_zarr(model, scenario, self.config.variable_name,
                                           start_analysis=start_time,
                                           end_analysis=end_time)
        
        return scenario_data
    
    def regrid_data(self, data: Dict[str, Dict[str, xr.DataArray]], 
                   reference_model: str = 'NorESM2-LM') -> Dict[str, Dict[str, xr.DataArray]]:
        """Regrid all data to a common grid."""
        print("Regridding data to common grid...")
        print("-" * 60)
        
        # Set reference grid (use the first main scenario data from reference model)
        first_scenario = list(data.keys())[0]
        self.reference_grid = data[first_scenario][reference_model]
        print(f"Using {reference_model} grid as reference")
        
        regridded_data = {}
        
        for scenario_name, scenario_data in data.items():
            print(f"  - Regridding {scenario_name}...")
            regridded_data[scenario_name] = {}
            
            for model_name, model_data in scenario_data.items():
                if model_name == reference_model:
                    # No regridding needed for reference model
                    regridded_data[scenario_name][model_name] = model_data
                else:
                    # Regrid to reference grid
                    regridded_data[scenario_name][model_name] = model_data.interp(
                        lat=self.reference_grid.lat, 
                        lon=self.reference_grid.lon
                    )
        
        print("✓ Regridding complete")
        return regridded_data


# =============================================================================
# BASE ANALYZER CLASS
# =============================================================================

class FireWeatherAnalyzer(ABC):
    """Abstract base class for fire weather analysis."""
    
    def __init__(self, config: FireWeatherAnalysisConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.plotting_manager = PlottingManager(config)
        
        # Ensure output_dir is a Path object and create directory
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.raw_data = None
        self.processed_data = {}
        
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 80)
        print(f"{self.config.analysis_type.upper()} ANALYSIS")
        print("=" * 80)
        
        # Load and preprocess data
        self.raw_data = self.data_manager.load_all_data()
        self.raw_data = self.data_manager.regrid_data(self.raw_data)
        
        # Process data (analysis-specific)
        self.processed_data = self.process_data(self.raw_data)
        
        # Run annual analysis
        if self.config.include_annual:
            self.run_annual_analysis()
        
        # Run seasonal analysis
        if self.config.include_seasonal:
            self.run_seasonal_analysis()
        
        self.print_summary()
    
    @abstractmethod
    def process_data(self, raw_data: Dict[str, Dict[str, xr.DataArray]]) -> Dict[str, Any]:
        """Process raw data for specific analysis type. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run_annual_analysis(self):
        """Run annual analysis. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run_seasonal_analysis(self):
        """Run seasonal analysis. Must be implemented by subclasses."""
        pass
    
    def combine_models(self, scenario_data: Dict[str, xr.DataArray]) -> xr.DataArray:
        """Combine data across models."""
        model_list = [scenario_data[model] for model in self.config.models]
        combined = xr.concat(model_list, dim="model", coords='minimal')
        return combined.compute()
    
    def calculate_anomalies(self, future_data: xr.DataArray, 
                          baseline_data: xr.DataArray) -> xr.DataArray:
        """Calculate anomalies relative to baseline."""
        return future_data - baseline_data
    
    def apply_masks_for_plotting(self, data: xr.DataArray, 
                                for_hatching: bool = False) -> Tuple[xr.DataArray, xr.DataArray]:
        """Apply masks for plotting (both filled contours and hatching)."""
        threshold = (self.config.agreement_threshold_hatching if for_hatching 
                    else self.config.agreement_threshold_plot)
        threshold_type = 'maximum' if for_hatching else 'minimum'
        
        masked_data, _ = apply_masks(data,
                                   get_significance=True,
                                   agreement_threshold=threshold,
                                   threshold_type=threshold_type,
                                   get_land_mask=True,
                                   baseline_data=data)
        
        return masked_data
    
    def calculate_global_average(self, data: xr.DataArray) -> xr.DataArray:
        """Calculate area-weighted global average."""
        masked_data, _ = apply_masks(data, get_significance=False, get_land_mask=True, baseline_data=None)
        # Always average across models first, then calculate weighted average
        if "model" in masked_data.dims:
            model_averaged = masked_data.mean('model')
            return weighted_horizontal_avg(model_averaged, ensemble=False, time=False)
        elif "member" in masked_data.dims:
            member_averaged = masked_data.mean('member')
            return weighted_horizontal_avg(member_averaged, ensemble=False, time=False)
        else:
            return weighted_horizontal_avg(masked_data, ensemble=False, time=False)
    
    def print_summary(self):
        """Print analysis summary."""
        print("=" * 80)
        print(f"{self.config.analysis_type.upper()} ANALYSIS - COMPLETE")
        print("=" * 80)
        
        plot_files = list(self.config.output_dir.glob('*.png'))
        annual_plots = len([f for f in plot_files if 'annual' in f.name])
        seasonal_plots = len([f for f in plot_files if 'seasonal' in f.name])
        
        print(f"✓ Annual analysis: {annual_plots} plots generated")
        print(f"✓ Seasonal analysis: {seasonal_plots} plots generated") 
        print(f"✓ Total plots: {len(plot_files)} plots generated")
        print(f"✓ All plots saved to: {self.config.output_dir}")
        print("=" * 80)


# =============================================================================
# PLOTTING MANAGER
# =============================================================================

class PlottingManager:
    """Handles all plotting and visualization operations."""
    
    def __init__(self, config: FireWeatherAnalysisConfig):
        self.config = config
    
    def create_historical_plot(self, data: xr.DataArray, 
                             filename: str, title: str = None) -> None:
        """Create historical baseline plot."""
        plot_params = self.config.plot_params['historical']
        
        if title is None:
            title = f"Historical ({self.config.historical_start.year}-{self.config.historical_end.year})"
        
        global_avg = self.calculate_global_average(data)
        
        # Apply land mask for map display
        masked_data, _ = apply_masks(data, get_significance=False, get_land_mask=True, baseline_data=None)
        
        fig, ax = create_global_map(
            masked_data.mean('model'),
            projection=ccrs.Robinson(),
            title=title,
            colormap=plot_params['colormap'],
            colorbar_title=plot_params['colorbar_title'],
            textbox_text=f"{global_avg.values.item():.2f}",
            figsize=(10.5, 6),
            vmin=plot_params['vmin'],
            vmax=plot_params['vmax'],
            extend='max',
            colorbar_levels=plot_params['colorbar_levels'],
            regional_boundaries='ar6',
            show_gridlines=False
        )
        
        plt.savefig(self.config.output_dir / filename, 
                   dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def create_main_scenarios_grid(self, data_dict: Dict[str, xr.DataArray],
                                 global_avgs: Dict[str, xr.DataArray],
                                 hatching_dict: Dict[str, xr.DataArray],
                                 filename: str, title: str = None) -> None:
        """Create grid plot for main scenarios."""
        plot_params = self.config.plot_params['main_scenarios']
        
        if title is None:
            title = f"{self.config.analysis_type.title()} Changes"
        
        # Prepare data for grid
        scenario_order = ['ssp370', 'ssp370-126aer', 'ssp126', 'aer126eff', 'ghg126eff']
        scenario_titles = ['SSP3-7.0', 'SSP3-7.0 with Global Aerosol Reduction', 'SSP1-2.6',
                          'Effect of Aerosol Emission Reduction', 'Effect of GHG Emission Reduction']
        
        data_list = [data_dict[scenario].mean('model') for scenario in scenario_order]
        textbox_texts = [f"{global_avgs[scenario].values.item():.2f}" for scenario in scenario_order]
        hatching_list = [hatching_dict[scenario].isel(model=0) for scenario in scenario_order]
        
        fig, axes = create_global_map_grid(
            data_list,
            rows=2, cols=3,
            main_title=title,
            projection=ccrs.Robinson(),
            titles=scenario_titles,
            colormaps=plot_params['colormap'],
            colorbar_titles=plot_params['colorbar_title'],
            textbox_texts=textbox_texts,
            vmins=plot_params['vmins'],
            vmaxs=plot_params['vmaxs'],
            extends='both',
            colorbar_levels=plot_params['colorbar_levels'],
            hatchings='///',
            regional_boundaries='ar6',
            hatching_styles='overlay',
            hatching_data=hatching_list,
            show_gridlines=False
        )
        
        plt.savefig(self.config.output_dir / filename,
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    def create_regional_scenarios_grid(self, data_dict: Dict[str, xr.DataArray],
                                     global_avgs: Dict[str, xr.DataArray],
                                     hatching_dict: Dict[str, xr.DataArray],
                                     filename: str, title: str = None) -> None:
        """Create grid plot for regional scenarios."""
        plot_params = self.config.plot_params['regional_scenarios']
        
        if title is None:
            title = f"Regional Aerosol Reduction Effects on {self.config.analysis_type.title()}"
        
        # Prepare data for grid
        data_list = [data_dict[scenario].mean('model') for scenario in self.config.regional_scenarios]
        textbox_texts = [f"{global_avgs[scenario].values.item():.2f}" 
                        for scenario in self.config.regional_scenarios]
        hatching_list = [hatching_dict[scenario].isel(model=0) 
                        for scenario in self.config.regional_scenarios]
        
        # Regional titles with proper names
        regional_titles = [f'Effect of {name} Aerosol Reduction' 
                          for name in self.config.regional_names]
        
        # RAMIP regions mapping
        ramip_regions = ['east_asia', 'north_america_europe', 'south_asia', 'africa_mideast']
        
        fig, axes = create_global_map_grid(
            data_list,
            rows=2, cols=2,
            main_title=title,
            projection=ccrs.Robinson(),
            titles=regional_titles,
            colormaps=plot_params['colormap'],
            colorbar_titles=plot_params['colorbar_title'],
            textbox_texts=textbox_texts,
            vmins=plot_params['vmins'],
            vmaxs=plot_params['vmaxs'],
            extends='both',
            colorbar_levels=plot_params['colorbar_levels'],
            hatchings='///',
            regional_boundaries='ar6',
            hatching_styles='overlay',
            hatching_data=hatching_list,
            show_gridlines=False,
            ramip_regions=ramip_regions
        )
        
        plt.savefig(self.config.output_dir / filename,
                   dpi=600, bbox_inches='tight')
        plt.close()
    
    def calculate_global_average(self, data: xr.DataArray) -> xr.DataArray:
        """Calculate area-weighted global average."""
        masked_data, _ = apply_masks(data, get_significance=False, get_land_mask=True, baseline_data=None)
        # Always average across models first, then calculate weighted average
        if "model" in masked_data.dims:
            model_averaged = masked_data.mean('model')
            return weighted_horizontal_avg(model_averaged, ensemble=False, time=False)
        elif "member" in masked_data.dims:
            member_averaged = masked_data.mean('member')
            return weighted_horizontal_avg(member_averaged, ensemble=False, time=False)
        else:
            return weighted_horizontal_avg(masked_data, ensemble=False, time=False)


if __name__ == "__main__":
    # Example usage will be shown in separate implementation files
    print("Fire Weather Analysis Framework loaded successfully!")
    print("Use specific analyzer classes (RawValueAnalyzer, AbsoluteThresholdAnalyzer, etc.) to run analyses.")
