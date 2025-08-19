#!/usr/bin/env python3
"""
Example: Threshold-Based Fire Weather Analysis
=============================================

This script demonstrates how to use the modular fire weather analysis framework
for threshold-based analyses:

1. Absolute threshold analysis (high fire danger days, FWI ≥ 20)
2. Percentile threshold analysis (days above 95th percentile)

These examples show how the same framework can be easily adapted for different
types of fire weather metrics.
"""

from fwi_analysis_framework import FireWeatherAnalysisConfig
from fwi_analyzers import AbsoluteThresholdAnalyzer, PercentileThresholdAnalyzer

def run_absolute_threshold_analysis():
    """Run high fire danger days analysis (FWI > 30)."""
    
    print("ABSOLUTE THRESHOLD ANALYSIS")
    print("=" * 50)
    
    # Create configuration for absolute threshold analysis
    config = FireWeatherAnalysisConfig.for_absolute_threshold(
        threshold=30,  # High fire danger threshold
        output_dir="high_fire_danger_plots",
        output_prefix="hfd",
        dpi=300
    )
    
    print("Configuration:")
    print(f"  Analysis type: {config.analysis_type}")
    print(f"  Threshold: FWI > {config.threshold_value}")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    # Create and run analyzer
    analyzer = AbsoluteThresholdAnalyzer(config)
    analyzer.run_analysis()

def run_percentile_threshold_analysis():
    """Run 95th percentile exceedance analysis."""
    
    print("\nPERCENTILE THRESHOLD ANALYSIS")
    print("=" * 50)
    
    # Create configuration for percentile threshold analysis
    config = FireWeatherAnalysisConfig.for_percentile_threshold(
        percentile=95,  # 95th percentile
        output_dir="percentile_95_plots",
        output_prefix="p95",
        dpi=300
    )
    
    print("Configuration:")
    print(f"  Analysis type: {config.analysis_type}")
    print(f"  Threshold: {config.threshold_value}th percentile")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    # Create and run analyzer
    analyzer = PercentileThresholdAnalyzer(config)
    analyzer.run_analysis()

def run_custom_analysis():
    """Example of customizing the analysis configuration."""
    
    print("\nCUSTOM CONFIGURATION EXAMPLE")
    print("=" * 50)
    
    # Create a custom configuration
    config = FireWeatherAnalysisConfig.for_absolute_threshold(
        threshold=40,  # Very high fire danger threshold
        output_dir="very_high_fire_danger_plots",
        output_prefix="vhfd",
        
        # Custom time periods
        historical_start=cftime.DatetimeNoLeap(1981, 1, 1, 12, 0, 0, 0, has_year_zero=True),
        historical_end=cftime.DatetimeNoLeap(2010, 12, 31, 12, 0, 0, 0, has_year_zero=True),
        future_start=cftime.DatetimeNoLeap(2071, 1, 1, 12, 0, 0, 0, has_year_zero=True),
        future_end=cftime.DatetimeNoLeap(2100, 12, 31, 12, 0, 0, 0, has_year_zero=True),
        
        # Custom models (subset)
        models=['NorESM2-LM', 'SPEAR'],
        
        # Only annual analysis
        include_seasonal=False,
        
        # Different statistical thresholds
        agreement_threshold_plot=0.75,
        agreement_threshold_hatching=0.8
    )
    
    print("Custom Configuration:")
    print(f"  Threshold: FWI > {config.threshold_value}")
    print(f"  Historical: {config.historical_start.year}-{config.historical_end.year}")
    print(f"  Future: {config.future_start.year}-{config.future_end.year}")
    print(f"  Models: {config.models}")
    print(f"  Include seasonal: {config.include_seasonal}")
    print()
    
    # Create and run analyzer
    analyzer = AbsoluteThresholdAnalyzer(config)
    analyzer.run_analysis()

def main():
    """Run all example analyses."""
    
    # Import here to avoid issues if not available during development
    import cftime
    
    print("FIRE WEATHER THRESHOLD ANALYSIS EXAMPLES")
    print("=" * 80)
    print()
    
    # Run different types of analyses
    run_absolute_threshold_analysis()
    run_percentile_threshold_analysis()
    run_custom_analysis()
    
    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
