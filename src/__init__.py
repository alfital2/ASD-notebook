"""
ASD Eye-Tracking Analysis package.
"""

# Import key functions and classes for easy access
from src.preprocessing import (
    preprocess_eye_tracking_data,
    add_helper_features,
    enforce_binocular_validity,
    automatic_binocular_calibration,
    remove_extreme_disparities,
    detect_and_handle_blinks,
    preprocess_pupil_size
)

from src.features import (
    extract_basic_features, 
    extract_temporal_features,
    create_feature_matrix,
    ROIAnalyzer,
    extract_combined_features
)

from src.quality import (
    calculate_comprehensive_quality_metrics,
    calculate_quality_trends,
    assess_data_quality
)

from src.visualization import (
    plot_gaze_trajectory,
    plot_disparity_analysis,
    plot_preprocessing_summary,
    plot_comprehensive_quality_trends,
    plot_social_attention_summary
)

from src.utils import (
    explore_data_structure,
    save_preprocessing_report,
    batch_preprocess_files,
    process_with_social_attention
)

# Set up configuration parameters (can be overridden)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1024
SAMPLING_RATE = 500
DATA_DIR = 'files/'
OUTPUT_DIR = 'output/'
FIGURE_DIR = 'figures/'

# Ensure necessary directories exist
import os
for directory in [OUTPUT_DIR, FIGURE_DIR]:
    os.makedirs(directory, exist_ok=True)