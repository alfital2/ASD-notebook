"""
Example script demonstrating how to import and use the ASD analysis modules.

This script shows how to:
1. Import functions from the src package
2. Process a single eye-tracking file
3. Perform quality assessment
4. Extract features
5. Generate visualizations

Usage:
$ python import_example.py
"""

import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

# Import functionality from the src package
from src import (
    # Preprocessing
    preprocess_eye_tracking_data,
    
    # Quality assessment
    calculate_comprehensive_quality_metrics,
    plot_comprehensive_quality_trends,
    
    # Feature extraction
    extract_basic_features,
    
    # Visualization
    plot_gaze_trajectory,
    plot_preprocessing_summary,
    
    # Utilities
    explore_data_structure
)

def main():
    """Main function demonstrating the ASD analysis workflow."""
    
    # Configuration
    data_dir = 'files/'
    output_dir = 'output/'
    figure_dir = 'figures/'
    
    # Create output directories if they don't exist
    for directory in [output_dir, figure_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Check for data files
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        print("⚠️ No CSV files found in the data directory.")
        return
    
    # Process the first available file
    sample_file = csv_files[0]
    print(f"Processing file: {sample_file}")
    
    # Step 1: Explore data structure
    df_sample = explore_data_structure(sample_file)
    
    # Step 2: Preprocess the data
    df_processed, preprocessing_info = preprocess_eye_tracking_data(
        sample_file,
        verbose=True,
        create_plots=True
    )
    
    # Step 3: Save processed data
    subject_id = preprocessing_info['subject_id']
    output_file = os.path.join(output_dir, f"{subject_id}_processed.csv")
    df_processed.to_csv(output_file, index=False)
    print(f"✓ Processed data saved to: {output_file}")
    
    # Step 4: Assess data quality
    quality_metrics = calculate_comprehensive_quality_metrics(df_processed)
    fig = plot_comprehensive_quality_trends(quality_metrics)
    fig.savefig(os.path.join(figure_dir, f"{subject_id}_quality_assessment.png"))
    plt.close(fig)
    print(f"✓ Quality assessment saved to: {figure_dir}/{subject_id}_quality_assessment.png")
    
    # Step 5: Extract features
    features = extract_basic_features(df_processed)
    
    # Display some key features
    print("\n📊 Extracted Features (sample):")
    for key, value in list(features.items())[:5]:
        print(f"   {key}: {value:.3f}")
    
    # Save features
    features_df = pd.DataFrame([features])
    features_file = os.path.join(output_dir, f"{subject_id}_features.csv")
    features_df.to_csv(features_file, index=False)
    print(f"✓ Features saved to: {features_file}")
    
    print("\n✅ Processing complete!")
    print(f"Data retention: {preprocessing_info['retention_rate']:.1f}%")

if __name__ == "__main__":
    main()