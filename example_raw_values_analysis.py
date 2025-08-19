#!/usr/bin/env python3
"""
Example: Raw FWI Values Analysis
==============================

This script demonstrates how to use the modular fire weather analysis framework
to perform the same analysis as the original mm_fwi_annual_seasonal_analysis.py
but with a cleaner, more modular approach.

This example replicates the exact analysis from your original script but with
much less code and better organization.
"""

from fwi_analysis_framework import FireWeatherAnalysisConfig
from fwi_analyzers import RawValueAnalyzer

def main():
    """Run raw FWI values analysis."""
    
    # Create configuration for raw FWI analysis
    config = FireWeatherAnalysisConfig.for_raw_values(
        output_dir="mm_fwi_plots_modular",
        output_prefix="mm",
        dpi=300
    )
    
    print("Configuration:")
    print(f"  Analysis type: {config.analysis_type}")
    print(f"  Models: {config.models}")
    print(f"  Historical period: {config.historical_start.year}-{config.historical_end.year}")
    print(f"  Future period: {config.future_start.year}-{config.future_end.year}")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    # Create and run analyzer
    analyzer = RawValueAnalyzer(config)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
