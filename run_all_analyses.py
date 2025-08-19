#!/usr/bin/env python3
"""
Complete Fire Weather Analysis Suite
===================================

This script demonstrates running multiple types of fire weather analyses
using the modular framework. It shows how easy it is to perform comprehensive
fire weather research with minimal code.

This replaces the need for multiple separate analysis scripts and ensures
consistent methodology across all analysis types.
"""

import cftime
from pathlib import Path

from fwi_analysis_framework import FireWeatherAnalysisConfig
from fwi_analyzers import RawValueAnalyzer, AbsoluteThresholdAnalyzer, PercentileThresholdAnalyzer

def run_comprehensive_fire_weather_analysis():
    """Run a comprehensive suite of fire weather analyses."""
    
    print("=" * 80)
    print("COMPREHENSIVE FIRE WEATHER ANALYSIS SUITE")
    print("=" * 80)
    print()
    
    # Base output directory
    base_output_dir = Path("comprehensive_fire_weather_analysis")
    base_output_dir.mkdir(exist_ok=True)
    
    analyses_run = []
    
    # 1. RAW FWI VALUES ANALYSIS
    print("1. RAW FWI VALUES ANALYSIS")
    print("-" * 40)
    
    try:
        config_raw = FireWeatherAnalysisConfig.for_raw_values(
            output_dir=base_output_dir / "raw_fwi_values",
            output_prefix="raw_fwi"
        )
        
        analyzer_raw = RawValueAnalyzer(config_raw)
        analyzer_raw.run_analysis()
        analyses_run.append("Raw FWI Values")
        
    except Exception as e:
        print(f"Error in raw FWI analysis: {e}")
    
    print()
    
    # 2. HIGH FIRE DANGER DAYS (FWI > 30)
    print("2. HIGH FIRE DANGER DAYS ANALYSIS (FWI > 30)")
    print("-" * 49)
    
    try:
        config_hfd = FireWeatherAnalysisConfig.for_absolute_threshold(
            threshold=30,
            output_dir=base_output_dir / "high_fire_danger_days",
            output_prefix="hfd30"
        )
        
        analyzer_hfd = AbsoluteThresholdAnalyzer(config_hfd)
        analyzer_hfd.run_analysis()
        analyses_run.append("High Fire Danger Days (FWI > 30)")
        
    except Exception as e:
        print(f"Error in high fire danger analysis: {e}")
    
    print()
    
    # 3. EXTREME FIRE DANGER DAYS (FWI > 40)
    print("3. EXTREME FIRE DANGER DAYS ANALYSIS (FWI > 40)")
    print("-" * 52)
    
    try:
        config_efd = FireWeatherAnalysisConfig.for_absolute_threshold(
            threshold=40,
            output_dir=base_output_dir / "extreme_fire_danger_days",
            output_prefix="efd40"
        )
        
        analyzer_efd = AbsoluteThresholdAnalyzer(config_efd)
        analyzer_efd.run_analysis()
        analyses_run.append("Extreme Fire Danger Days (FWI > 40)")
        
    except Exception as e:
        print(f"Error in extreme fire danger analysis: {e}")
    
    print()
    
    # 4. 95TH PERCENTILE EXCEEDANCE
    print("4. 95TH PERCENTILE EXCEEDANCE ANALYSIS")
    print("-" * 43)
    
    try:
        config_p95 = FireWeatherAnalysisConfig.for_percentile_threshold(
            percentile=95,
            output_dir=base_output_dir / "percentile_95_exceedance",
            output_prefix="p95"
        )
        
        analyzer_p95 = PercentileThresholdAnalyzer(config_p95)
        analyzer_p95.run_analysis()
        analyses_run.append("95th Percentile Exceedance")
        
    except Exception as e:
        print(f"Error in 95th percentile analysis: {e}")
    
    print()
    
    # 5. 99TH PERCENTILE EXCEEDANCE
    print("5. 99TH PERCENTILE EXCEEDANCE ANALYSIS")
    print("-" * 43)
    
    try:
        config_p99 = FireWeatherAnalysisConfig.for_percentile_threshold(
            percentile=99,
            output_dir=base_output_dir / "percentile_99_exceedance",
            output_prefix="p99"
        )
        
        analyzer_p99 = PercentileThresholdAnalyzer(config_p99)
        analyzer_p99.run_analysis()
        analyses_run.append("99th Percentile Exceedance")
        
    except Exception as e:
        print(f"Error in 99th percentile analysis: {e}")
    
    print()
    
    # Summary
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Successfully completed {len(analyses_run)} analyses:")
    for i, analysis in enumerate(analyses_run, 1):
        print(f"  {i}. {analysis}")
    
    print(f"\nAll results saved to: {base_output_dir}")
    print(f"Total output directories: {len(list(base_output_dir.iterdir()))}")
    
    # Count total plots generated
    total_plots = 0
    for subdir in base_output_dir.iterdir():
        if subdir.is_dir():
            plot_count = len(list(subdir.glob("*.png")))
            total_plots += plot_count
            print(f"  {subdir.name}: {plot_count} plots")
    
    print(f"\nTotal plots generated: {total_plots}")
    print("=" * 80)

def run_quick_comparison():
    """Run a quick comparison with different time periods."""
    
    print("\nQUICK COMPARISON: DIFFERENT TIME PERIODS")
    print("=" * 80)
    
    # Current vs. future periods
    periods = [
        {
            'name': 'Early_21st_Century',
            'hist_start': cftime.DatetimeNoLeap(1995, 1, 1, 12, 0, 0, 0, has_year_zero=True),
            'hist_end': cftime.DatetimeNoLeap(2014, 12, 31, 12, 0, 0, 0, has_year_zero=True),
            'fut_start': cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0, 0, has_year_zero=True),
            'fut_end': cftime.DatetimeNoLeap(2034, 12, 31, 12, 0, 0, 0, has_year_zero=True)
        },
        {
            'name': 'Mid_21st_Century',
            'hist_start': cftime.DatetimeNoLeap(1961, 1, 1, 12, 0, 0, 0, has_year_zero=True),
            'hist_end': cftime.DatetimeNoLeap(1990, 12, 31, 12, 0, 0, 0, has_year_zero=True),
            'fut_start': cftime.DatetimeNoLeap(2041, 1, 1, 12, 0, 0, 0, has_year_zero=True),
            'fut_end': cftime.DatetimeNoLeap(2070, 12, 31, 12, 0, 0, 0, has_year_zero=True)
        }
    ]
    
    for period in periods:
        print(f"\nAnalyzing period: {period['name']}")
        print(f"Historical: {period['hist_start'].year}-{period['hist_end'].year}")
        print(f"Future: {period['fut_start'].year}-{period['fut_end'].year}")
        
        try:
            config = FireWeatherAnalysisConfig.for_raw_values(
                historical_start=period['hist_start'],
                historical_end=period['hist_end'],
                future_start=period['fut_start'],
                future_end=period['fut_end'],
                output_dir=f"comparison_{period['name']}",
                output_prefix=period['name'].lower(),
                include_seasonal=False  # Annual only for quick comparison
            )
            
            analyzer = RawValueAnalyzer(config)
            analyzer.run_analysis()
            
        except Exception as e:
            print(f"Error in {period['name']} analysis: {e}")

def main():
    """Main execution function."""
    
    # Check if we're in the right directory
    if not Path("ramip_fwi_utilities.py").exists():
        print("Error: Please run this script from the fire_weather directory")
        print("The script needs access to ramip_fwi_utilities.py")
        return
    
    # Run comprehensive analysis
    run_comprehensive_fire_weather_analysis()
    
    # Optionally run quick comparison (commented out to avoid long runtime)
    # run_quick_comparison()

if __name__ == "__main__":
    main()
