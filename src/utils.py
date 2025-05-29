"""
Utility functions for eye-tracking data analysis.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Union

# Configuration
OUTPUT_DIR = 'output/'
FIGURE_DIR = 'figures/'


def explore_data_structure(file_path: str) -> pd.DataFrame:
    """
    Load and explore the structure of an eye-tracking CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract subject ID from filename
    subject_id = os.path.basename(file_path).split('_')[0]
    
    print(f"📊 Data Structure Analysis")
    print(f"{'='*50}")
    print(f"Subject ID: {subject_id}")
    print(f"Total samples: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    # Check recording duration
    if 'timestamp' in df.columns:
        duration_ms = df['timestamp'].max() - df['timestamp'].min()
        duration_sec = duration_ms / 1000
        print(f"Recording duration: {duration_sec:.2f} seconds")
        print(f"Effective sampling rate: {len(df) / duration_sec:.2f} Hz")
    
    # Analyze missing data
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    
    print(f"\n📋 Column Information:")
    print(f"{'Column':<25} {'Type':<10} {'Missing %':<10} {'Description'}")
    print(f"{'-'*70}")
    
    column_descriptions = {
        'timestamp': 'Recording timestamp (ms)',
        'frame_number': 'Video frame number',
        'x_left': 'Left eye X coordinate',
        'y_left': 'Left eye Y coordinate',
        'x_right': 'Right eye X coordinate',
        'y_right': 'Right eye Y coordinate',
        'pupil_left': 'Left pupil diameter',
        'pupil_right': 'Right pupil diameter',
        'is_fixation_left': 'Left eye fixation flag',
        'is_fixation_right': 'Right eye fixation flag',
        'is_saccade_left': 'Left eye saccade flag',
        'is_saccade_right': 'Right eye saccade flag',
        'is_blink_left': 'Left eye blink flag',
        'is_blink_right': 'Right eye blink flag',
    }
    
    for col in df.columns:
        desc = column_descriptions.get(col, 'Additional metric')
        print(f"{col:<25} {str(df[col].dtype):<10} {missing_pct[col]:<10.1f} {desc}")
    
    return df


def save_preprocessing_report(
    preprocessing_info: Dict,
    df_processed: pd.DataFrame,
    output_dir: str = None
) -> str:
    """
    Generate and save a comprehensive preprocessing report.
    
    Args:
        preprocessing_info: Dictionary with preprocessing statistics
        df_processed: Processed dataframe
        output_dir: Directory to save report (default: OUTPUT_DIR)
        
    Returns:
        report_path: Path to saved report
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    subject_id = preprocessing_info['subject_id']
    report_path = os.path.join(output_dir, f"{subject_id}_preprocessing_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("EYE-TRACKING PREPROCESSING REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Basic information
        f.write(f"Subject ID: {subject_id}\n")
        f.write(f"File: {preprocessing_info['file']}\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Data retention summary
        f.write("DATA RETENTION SUMMARY\n")
        f.write("-"*30 + "\n")
        f.write(f"Initial samples: {preprocessing_info['initial_samples']:,}\n")
        f.write(f"Final valid samples: {preprocessing_info['final_valid_samples']:,}\n")
        f.write(f"Retention rate: {preprocessing_info['retention_rate']:.1f}%\n")
        f.write("\n")
        
        # Pipeline steps
        f.write("PREPROCESSING PIPELINE\n")
        f.write("-"*30 + "\n")
        for step, samples in preprocessing_info['steps'].items():
            f.write(f"{step}: {samples:,} samples\n")
        f.write("\n")
        
        # Blink statistics
        f.write("BLINK DETECTION\n")
        f.write("-"*30 + "\n")
        for eye, stats in preprocessing_info['blink_stats'].items():
            f.write(f"{eye}:\n")
            f.write(f"  Natural blinks: {stats['natural_blinks']}\n")
            f.write(f"  Technical gaps: {stats['technical_gaps']}\n")
            f.write(f"  Interpolated samples: {stats['interpolated_samples']}\n")
        f.write("\n")
        
        # Calibration correction
        f.write("BINOCULAR CALIBRATION\n")
        f.write("-"*30 + "\n")
        cal = preprocessing_info['calibration_correction']
        f.write(f"Improvement: {cal['improvement_percent']:.1f}%\n")
        f.write(f"Original disparity: {cal.get('original_disparity', 0):.1f} pixels\n")
        f.write(f"Corrected disparity: {cal.get('corrected_disparity', 0):.1f} pixels\n")
        f.write("\n")
        
        # Pupil preprocessing
        f.write("PUPIL PREPROCESSING\n")
        f.write("-"*30 + "\n")
        for eye, stats in preprocessing_info['pupil_stats'].items():
            f.write(f"{eye}:\n")
            f.write(f"  Removed: {stats['removal_percentage']:.1f}%\n")
            f.write(f"  Final valid: {stats['final_valid']:,} samples\n")
        f.write("\n")
        
        # Feature summary
        f.write("FEATURE SUMMARY\n")
        f.write("-"*30 + "\n")
        from src.features import extract_basic_features
        features = extract_basic_features(df_processed)
        for key, value in sorted(features.items()):
            if isinstance(value, float):
                f.write(f"{key}: {value:.3f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"✅ Report saved to: {report_path}")
    return report_path


def batch_preprocess_files(
    file_pattern: str = None,
    output_dir: str = None,
    create_plots: bool = False,
    save_preprocessed: bool = True
) -> pd.DataFrame:
    """
    Batch process multiple eye-tracking files.
    
    Args:
        file_pattern: Glob pattern for files (default: all CSVs in DATA_DIR)
        output_dir: Directory to save processed files (default: OUTPUT_DIR)
        create_plots: Generate plots for each file
        save_preprocessed: Save preprocessed data to CSV

    Returns:
        summary_df: DataFrame with preprocessing summary for all files
    """
    from src.preprocessing import preprocess_eye_tracking_data
    
    # Use defaults from config
    DATA_DIR = 'files/'
    if file_pattern is None:
        file_pattern = os.path.join(DATA_DIR, '*.csv')
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Get all files
    files = sorted(glob.glob(file_pattern))
    print(f"Found {len(files)} files to process")

    # Process each file
    results = []

    for i, file_path in enumerate(files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(files)}")

        try:
            # Process file
            df_processed, info = preprocess_eye_tracking_data(
                file_path,
                verbose=True,
                create_plots=create_plots
            )

            # Save processed data
            if save_preprocessed:
                output_file = os.path.join(
                    output_dir,
                    f"{info['subject_id']}_preprocessed.csv"
                )
                df_processed.to_csv(output_file, index=False)
                print(f"✓ Saved to: {output_file}")

            # Add to results
            results.append({
                'subject_id': info['subject_id'],
                'file': info['file'],
                'initial_samples': info['initial_samples'],
                'final_samples': info['final_valid_samples'],
                'retention_rate': info['retention_rate'],
                'blinks_left': info['blink_stats']['left_eye']['natural_blinks'],
                'blinks_right': info['blink_stats']['right_eye']['natural_blinks'],
                'calibration_improvement': info['calibration_correction']['improvement_percent'],
                'pupil_removed_left': info['pupil_stats'].get('left_eye', {}).get('removal_percentage', 0),
                'pupil_removed_right': info['pupil_stats'].get('right_eye', {}).get('removal_percentage', 0),
            })

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            results.append({
                'subject_id': os.path.basename(file_path).split('_')[0],
                'file': os.path.basename(file_path),
                'error': str(e)
            })

    # Create summary dataframe
    summary_df = pd.DataFrame(results)

    # Save summary
    summary_file = os.path.join(output_dir, 'preprocessing_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Batch processing complete!")
    print(f"   Processed: {len(files)} files")
    print(f"   Average retention: {summary_df['retention_rate'].mean():.1f}%")
    print(f"   Summary saved to: {summary_file}")

    return summary_df


def create_preprocessing_module() -> None:
    """
    Create a standalone Python module with all preprocessing functions.
    """
    module_path = os.path.join(OUTPUT_DIR, 'asd_preprocessing.py')
    
    # Write the module
    with open(module_path, 'w') as f:
        f.write('"""\nASD Eye-Tracking Preprocessing Module\n')
        f.write('Generated from asd_professional.ipynb\n"""\n\n')
        
        # Add imports
        f.write('import pandas as pd\n')
        f.write('import numpy as np\n')
        f.write('from typing import Dict, Tuple, List, Optional\n')
        f.write('from scipy.ndimage import gaussian_filter1d\n\n')
        
        # Add configuration
        f.write('# Configuration\n')
        f.write('SCREEN_WIDTH = 1280\n')
        f.write('SCREEN_HEIGHT = 1024\n')
        f.write('SAMPLING_RATE = 500\n')
        f.write('PREPROCESSING_PARAMS = {...}\n\n')
        
        # Note about functions
        f.write('# Note: Copy preprocessing functions from notebook\n')
        f.write('# Functions include:\n')
        f.write('# - add_helper_features()\n')
        f.write('# - remove_empty_columns()\n')
        f.write('# - enforce_binocular_validity()\n')
        f.write('# - automatic_binocular_calibration()\n')
        f.write('# - remove_extreme_disparities()\n')
        f.write('# - detect_and_handle_blinks()\n')
        f.write('# - preprocess_pupil_size()\n')
        f.write('# - round_coordinates_to_pixels()\n')
        f.write('# - preprocess_eye_tracking_data()\n')
        f.write('# - extract_basic_features()\n')
    
    print(f"✅ Module template created: {module_path}")
    print("   Copy preprocessing functions from notebook to complete the module")


def process_with_social_attention(file_path: str, roi_file_path: str, create_plots: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Process eye tracking data with social attention analysis.
    
    Args:
        file_path: Path to eye tracking CSV file
        roi_file_path: Path to ROI JSON file
        create_plots: Whether to generate and save visualization plots
        
    Returns:
        df_processed: Processed dataframe with ROI information
        metrics: Dictionary with all metrics including social attention
    """
    from src.preprocessing import preprocess_eye_tracking_data
    from src.features import ROIAnalyzer, extract_combined_features
    from src.visualization import (
        plot_social_attention_summary,
        plot_temporal_social_attention,
        visualize_gaze_on_roi
    )
    
    print(f"\n🔎 Processing with Social Attention Analysis")
    print(f"File: {os.path.basename(file_path)}")
    print("="*60)
    
    # Extract subject ID from filename
    subject_id = os.path.basename(file_path).split('_')[0]
    
    # Step 1: Standard preprocessing
    df_processed, preprocessing_info = preprocess_eye_tracking_data(
        file_path,
        verbose=True,
        create_plots=False  # Set to True if you want preprocessing plots
    )
    
    print(f"\n✅ Standard preprocessing complete")
    print(f"   Valid samples: {preprocessing_info['final_valid_samples']:,} "
          f"({preprocessing_info['retention_rate']:.1f}%)")
    
    # Step 2: Initialize ROI analyzer
    print(f"\n📊 Initializing ROI Analyzer")
    roi_analyzer = ROIAnalyzer(roi_file_path)
    
    # Step 3: Analyze social attention
    print(f"\n👁 Analyzing Social Attention Patterns")
    social_metrics = roi_analyzer.analyze_gaze_social_attention(df_processed)
    
    # Step 4: Extract all features (standard + social)
    features = extract_combined_features(df_processed, roi_analyzer)
    
    # Print summary
    print(f"\n📋 Social Attention Summary:")
    print(f"   Social attention: {social_metrics['social_attention_percent']:.1f}%")
    print(f"   Face attention: {social_metrics['face_attention_percent']:.1f}%")
    print(f"   Head attention: {social_metrics['head_attention_percent']:.1f}%")
    print(f"   Hand attention: {social_metrics['hand_attention_percent']:.1f}%")
    print(f"   Torso attention: {social_metrics['torso_attention_percent']:.1f}%")
    print(f"   Person attention: {social_metrics['person_attention_percent']:.1f}%")
    print(f"   Object attention: {social_metrics['object_attention_percent']:.1f}%")
    print(f"   Social-to-nonsocial transitions: {social_metrics['transitions']['social_to_nonsocial']}")
    print(f"   Nonsocial-to-social transitions: {social_metrics['transitions']['nonsocial_to_social']}")
    
    # Create and save visualizations if requested
    if create_plots:
        # Create subject directory for saving plots
        subject_dir = os.path.join(FIGURE_DIR, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        print(f"\n🎨 Creating social attention visualizations")
        
        # 1. Summary plot
        fig1 = plot_social_attention_summary(social_metrics)
        fig1_path = os.path.join(subject_dir, f"social_attention_summary.png")
        fig1.savefig(fig1_path, dpi=100)
        plt.close(fig1)
        print(f"   Social attention summary saved to: {fig1_path}")
        
        # 2. Temporal plot
        fig2 = plot_temporal_social_attention(social_metrics)
        fig2_path = os.path.join(subject_dir, f"temporal_social_attention.png")
        fig2.savefig(fig2_path, dpi=100)
        plt.close(fig2)
        print(f"   Temporal social attention saved to: {fig2_path}")
        
        # 3. Sample frame visualization (if frames with ROI data exist)
        if 'df_with_roi' in social_metrics:
            df_roi = social_metrics['df_with_roi']
            # Find a frame with ROI data
            roi_frames = [int(k) for k in roi_analyzer.roi_data.keys() if k.isdigit() and roi_analyzer.roi_data[k]]
            if roi_frames:
                sample_frame = roi_frames[len(roi_frames)//2]  # Middle frame
                fig3 = visualize_gaze_on_roi(df_roi, sample_frame, roi_analyzer.roi_data)
                fig3_path = os.path.join(subject_dir, f"gaze_roi_frame_{sample_frame}.png")
                fig3.savefig(fig3_path, dpi=100)
                plt.close(fig3)
                print(f"   Gaze-ROI visualization saved to: {fig3_path}")
    
    # Combine all metrics
    all_metrics = {
        'preprocessing': preprocessing_info,
        'social_attention': social_metrics,
        'features': features
    }
    
    print(f"\n✨ Processing complete with {len(features)} total features")
    
    return df_processed, all_metrics