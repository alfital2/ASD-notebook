"""
Preprocessing functions for eye-tracking data analysis.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import gaussian_filter1d

# Configuration (these can be overridden by importing code)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1024
SCREEN_MARGIN = 500  # pixels outside screen to consider invalid

# Recording parameters
SAMPLING_RATE = 500  # Hz
RECORDING_DURATION = 90  # seconds
EXPECTED_SAMPLES = SAMPLING_RATE * RECORDING_DURATION

# Default preprocessing parameters
PREPROCESSING_PARAMS = {
    # Blink detection
    'blink_gap_threshold': 5,      # samples to consider as gap
    'blink_window': 24,            # max samples for natural blink (~48ms)
    
    # Pupil validation
    'pupil_min': 200,              # minimum valid pupil size
    'pupil_max': 1200,             # maximum valid pupil size
    'pupil_smoothing_window': 250, # Gaussian smoothing window (500ms)
    'pupil_adaptation_time': 1.0,  # seconds to exclude at start
    
    # Binocular disparity
    'max_disparity': 150,          # maximum allowed disparity (pixels)
    'disparity_percentile': 50,    # percentile for calibration correction
    'velocity_threshold': 100,     # velocity threshold for stable periods
    
    # Missing data
    'missing_data_threshold': 0.95,# threshold to remove columns
    
    # Visualization
    'plot_sample_rate': 10,        # sample every N points for scatter plots
    'figure_dpi': 100,             # figure resolution
}


def add_helper_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add helper features for preprocessing and analysis.
    
    Args:
        df: Input dataframe
        
    Returns:
        df: DataFrame with added features
    """
    df = df.copy()
    
    # Time in seconds from start
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
    
    # Velocity calculation
    df['velocity_left'] = np.sqrt(
        df['x_left'].diff()**2 + df['y_left'].diff()**2
    ) * SAMPLING_RATE
    
    df['velocity_right'] = np.sqrt(
        df['x_right'].diff()**2 + df['y_right'].diff()**2
    ) * SAMPLING_RATE
    
    # Binocular disparity
    df['disparity_x'] = df['x_left'] - df['x_right']
    df['disparity_y'] = df['y_left'] - df['y_right']
    df['disparity_total'] = np.sqrt(
        df['disparity_x']**2 + df['disparity_y']**2
    )
    
    # Validity flags
    df['both_eyes_valid'] = (
        df['x_left'].notna() & df['x_right'].notna() &
        df['y_left'].notna() & df['y_right'].notna()
    )
    
    # Screen bounds check
    df['left_in_bounds'] = (
        (df['x_left'] >= -20) & (df['x_left'] <= SCREEN_WIDTH + 20) &
        (df['y_left'] >= -20) & (df['y_left'] <= SCREEN_HEIGHT + 20)
    )
    
    df['right_in_bounds'] = (
        (df['x_right'] >= -20) & (df['x_right'] <= SCREEN_WIDTH + 20) &
        (df['y_right'] >= -20) & (df['y_right'] <= SCREEN_HEIGHT + 20)
    )
    
    return df


def remove_empty_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove columns with more than threshold proportion of missing data.
    
    Args:
        df: Input dataframe
        threshold: Maximum allowed proportion of missing data
        
    Returns:
        df: DataFrame with empty columns removed
    """
    missing_proportion = df.isnull().sum() / len(df)
    cols_to_remove = missing_proportion[missing_proportion > threshold].index.tolist()
    
    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} columns with >{threshold*100:.0f}% missing data:")
        for col in cols_to_remove:
            print(f"  - {col}: {missing_proportion[col]*100:.1f}% missing")
    
    return df.drop(columns=cols_to_remove)


def enforce_binocular_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce binocular data validity: if one eye's data is invalid, 
    mark both eyes as invalid for that sample.
    
    Args:
        df: Input dataframe
        
    Returns:
        df: DataFrame with enforced binocular validity
    """
    df = df.copy()
    
    # Find samples where only one eye has valid data
    left_valid = df['x_left'].notna() & df['y_left'].notna()
    right_valid = df['x_right'].notna() & df['y_right'].notna()
    
    # Identify monocular samples
    monocular_samples = (left_valid & ~right_valid) | (~left_valid & right_valid)
    
    print(f"Found {monocular_samples.sum()} monocular samples ({monocular_samples.sum()/len(df)*100:.1f}%)")
    
    # Invalidate monocular samples
    df.loc[monocular_samples, ['x_left', 'y_left', 'x_right', 'y_right']] = np.nan
    df.loc[monocular_samples, ['pupil_left', 'pupil_right']] = np.nan
    df.loc[monocular_samples, ['is_fixation_left', 'is_fixation_right']] = False
    df.loc[monocular_samples, ['is_saccade_left', 'is_saccade_right']] = False
    
    # Update validity flag
    df['both_eyes_valid'] = (
        df['x_left'].notna() & df['x_right'].notna() &
        df['y_left'].notna() & df['y_right'].notna()
    )
    
    print(f"After enforcement: {df['both_eyes_valid'].sum()} valid binocular samples ({df['both_eyes_valid'].sum()/len(df)*100:.1f}%)")
    
    return df


def automatic_binocular_calibration(
    df: pd.DataFrame,
    velocity_threshold: float = None,
    disparity_percentile: float = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Automatically correct small alignment errors between left and right eyes.
    
    Args:
        df: Input dataframe
        velocity_threshold: Max velocity for stable periods (default from config)
        disparity_percentile: Percentile for bias estimation (default from config)
        verbose: Print calibration info
        
    Returns:
        df: Calibrated dataframe
        correction_info: Dictionary with calibration statistics
    """
    df = df.copy()
    
    # Use defaults from config if not provided
    if velocity_threshold is None:
        velocity_threshold = PREPROCESSING_PARAMS['velocity_threshold']
    if disparity_percentile is None:
        disparity_percentile = PREPROCESSING_PARAMS['disparity_percentile']
    
    # Ensure velocities are calculated
    if 'velocity_left' not in df.columns:
        df['velocity_left'] = np.sqrt(
            df['x_left'].diff()**2 + df['y_left'].diff()**2
        ) * SAMPLING_RATE
    
    if 'velocity_right' not in df.columns:
        df['velocity_right'] = np.sqrt(
            df['x_right'].diff()**2 + df['y_right'].diff()**2
        ) * SAMPLING_RATE
    
    # Find stable periods
    stable_mask = (
        (df['velocity_left'] < velocity_threshold) &
        (df['velocity_right'] < velocity_threshold) &
        df['both_eyes_valid']
    )
    
    stable_samples = stable_mask.sum()
    
    if stable_samples < 100:  # Need minimum samples for calibration
        if verbose:
            print("⚠️ Insufficient stable samples for calibration")
        return df, {'improvement_percent': 0}
    
    # Calculate disparities during stable periods
    stable_data = df[stable_mask]
    x_disparity = stable_data['x_left'] - stable_data['x_right']
    y_disparity = stable_data['y_left'] - stable_data['y_right']
    
    # Calculate robust bias estimates
    x_bias = np.percentile(x_disparity, disparity_percentile)
    y_bias = np.percentile(y_disparity, disparity_percentile)
    
    # Split correction between both eyes
    x_correction_left = x_bias / 2
    y_correction_left = y_bias / 2
    x_correction_right = -x_bias / 2
    y_correction_right = -y_bias / 2
    
    # Apply corrections
    df['x_left'] = df['x_left'] - x_correction_left
    df['y_left'] = df['y_left'] - y_correction_left
    df['x_right'] = df['x_right'] - x_correction_right
    df['y_right'] = df['y_right'] - y_correction_right
    
    # Recalculate disparities
    df['disparity_x'] = df['x_left'] - df['x_right']
    df['disparity_y'] = df['y_left'] - df['y_right']
    df['disparity_total'] = np.sqrt(
        df['disparity_x']**2 + df['disparity_y']**2
    )
    
    # Calculate improvement
    original_disparity = np.sqrt(x_disparity**2 + y_disparity**2).median()
    corrected_disparity = df.loc[stable_mask, 'disparity_total'].median()
    improvement = (1 - corrected_disparity / original_disparity) * 100
    
    correction_info = {
        'x_bias': x_bias,
        'y_bias': y_bias,
        'stable_samples': stable_samples,
        'original_disparity': original_disparity,
        'corrected_disparity': corrected_disparity,
        'improvement_percent': improvement,
    }
    
    if verbose:
        print("🔧 Automatic Binocular Calibration")
        print(f"  Detected bias: ({x_bias:.1f}, {y_bias:.1f}) pixels")
        print(f"  Based on {stable_samples:,} stable samples")
        print(f"  Disparity reduced: {original_disparity:.1f} → {corrected_disparity:.1f} pixels")
        print(f"  Improvement: {improvement:.1f}%")
    
    return df, correction_info


def remove_extreme_disparities(
    df: pd.DataFrame,
    max_disparity: float = None
) -> pd.DataFrame:
    """
    Remove samples with physiologically impossible binocular disparities.
    
    Args:
        df: Input dataframe
        max_disparity: Maximum allowed disparity (default from config)
        
    Returns:
        df: DataFrame with extreme disparities removed
    """
    df = df.copy()
    
    if max_disparity is None:
        max_disparity = PREPROCESSING_PARAMS['max_disparity']
    
    # Calculate disparity for valid samples
    both_valid = df['both_eyes_valid']
    
    # Find extreme disparities
    extreme_mask = both_valid & (df['disparity_total'] > max_disparity)
    n_extreme = extreme_mask.sum()
    
    print(f"Found {n_extreme} samples with extreme disparity (>{max_disparity} pixels)")
    print(f"That's {n_extreme/both_valid.sum()*100:.2f}% of previously valid data")
    
    # Invalidate extreme samples
    df.loc[extreme_mask, ['x_left', 'y_left', 'x_right', 'y_right']] = np.nan
    df.loc[extreme_mask, ['pupil_left', 'pupil_right']] = np.nan
    df.loc[extreme_mask, 'both_eyes_valid'] = False
    
    # Also check for out-of-bounds data
    out_of_bounds = both_valid & (
        (df['x_left'] < -SCREEN_MARGIN) | (df['x_left'] > SCREEN_WIDTH + SCREEN_MARGIN) |
        (df['y_left'] < -SCREEN_MARGIN) | (df['y_left'] > SCREEN_HEIGHT + SCREEN_MARGIN) |
        (df['x_right'] < -SCREEN_MARGIN) | (df['x_right'] > SCREEN_WIDTH + SCREEN_MARGIN) |
        (df['y_right'] < -SCREEN_MARGIN) | (df['y_right'] > SCREEN_HEIGHT + SCREEN_MARGIN)
    )
    n_oob = out_of_bounds.sum()
    
    if n_oob > 0:
        print(f"Found {n_oob} samples far outside screen bounds")
        df.loc[out_of_bounds, ['x_left', 'y_left', 'x_right', 'y_right']] = np.nan
        df.loc[out_of_bounds, 'both_eyes_valid'] = False
    
    # Update validity
    df['both_eyes_valid'] = (
        df['x_left'].notna() & df['x_right'].notna() &
        df['y_left'].notna() & df['y_right'].notna()
    )
    
    print(f"After filtering: {df['both_eyes_valid'].sum()} valid samples ({df['both_eyes_valid'].sum()/len(df)*100:.1f}%)")
    
    return df


def detect_and_stabilize_noisy_eye(
    df: pd.DataFrame,
    noise_threshold: float = 3.0,
    outlier_threshold: float = 3.0,
    smoothing_method: str = 'savgol',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Detect and stabilize noisy/shaky eye tracking data.
    Uses multiple approaches to identify and fix tracking artifacts.

    Args:
        df: Input dataframe
        noise_threshold: Ratio threshold to detect noisy eye
        outlier_threshold: Standard deviations for outlier detection
        smoothing_method: 'savgol', 'median', or 'kalman'
        verbose: Print stabilization info

    Returns:
        df: DataFrame with stabilized eye data
    """
    df = df.copy()

    # Calculate noise metrics for each eye
    def calculate_noise_metrics(x_col, y_col):
        # Method 1: Velocity variance (sudden changes)
        velocity = np.sqrt(df[x_col].diff()**2 + df[y_col].diff()**2)
        velocity_var = velocity.var()
        
        # Method 2: Acceleration variance (jerkiness)
        acceleration = velocity.diff().abs()
        accel_var = acceleration.var()
        
        # Method 3: High-frequency noise (second derivative)
        x_noise = df[x_col].diff().diff().abs().mean()
        y_noise = df[y_col].diff().diff().abs().mean()
        
        # Method 4: Outlier count (sudden large jumps)
        outliers = ((df[x_col].diff().abs() > df[x_col].diff().abs().quantile(0.95)) |
                   (df[y_col].diff().abs() > df[y_col].diff().abs().quantile(0.95))).sum()
        
        return {
            'velocity_var': velocity_var,
            'accel_var': accel_var,
            'noise_level': x_noise + y_noise,
            'outlier_count': outliers,
            'total_samples': len(df[x_col].dropna())
        }

    # Analyze both eyes
    left_metrics = calculate_noise_metrics('x_left', 'y_left')
    right_metrics = calculate_noise_metrics('x_right', 'y_right')
    
    # Determine which eye is noisier
    left_noise_score = (left_metrics['velocity_var'] + left_metrics['accel_var'] + 
                       left_metrics['noise_level'] + left_metrics['outlier_count'])
    right_noise_score = (right_metrics['velocity_var'] + right_metrics['accel_var'] + 
                        right_metrics['noise_level'] + right_metrics['outlier_count'])
    
    if verbose:
        print(f"👁️ Eye tracking quality analysis:")
        print(f"   Left eye noise score:  {left_noise_score:.2f}")
        print(f"   Right eye noise score: {right_noise_score:.2f}")
    
    # Identify noisy eye
    noisy_eye = None
    stable_eye = None
    noise_ratio = 1.0
    
    if left_noise_score > noise_threshold * right_noise_score:
        noisy_eye = 'left'
        stable_eye = 'right'
        noise_ratio = left_noise_score / right_noise_score
    elif right_noise_score > noise_threshold * left_noise_score:
        noisy_eye = 'right'
        stable_eye = 'left'
        noise_ratio = right_noise_score / left_noise_score
    
    if noisy_eye is None:
        if verbose:
            print(f"✓ Both eyes have similar tracking quality (ratio: {max(left_noise_score, right_noise_score) / min(left_noise_score, right_noise_score):.2f})")
        return df
    
    if verbose:
        print(f"⚠️ Detected noisy tracking in {noisy_eye} eye (noise ratio: {noise_ratio:.2f})")
        print(f"   Applying {smoothing_method} stabilization...")
    
    # Apply stabilization to noisy eye
    for coord in ['x', 'y']:
        noisy_col = f'{coord}_{noisy_eye}'
        stable_col = f'{coord}_{stable_eye}'
        
        if noisy_col not in df.columns or stable_col not in df.columns:
            continue
            
        # Get valid data
        valid_mask = df[noisy_col].notna() & df[stable_col].notna()
        if valid_mask.sum() < 50:
            continue
            
        noisy_data = df.loc[valid_mask, noisy_col].copy()
        stable_data = df.loc[valid_mask, stable_col].copy()
        
        # Method 1: Remove extreme outliers first
        median_val = noisy_data.median()
        mad = (noisy_data - median_val).abs().median()
        outlier_mask = (noisy_data - median_val).abs() > outlier_threshold * mad * 1.4826
        
        # Method 2: Apply different smoothing techniques
        if smoothing_method == 'savgol':
            # Savitzky-Golay filter - good for preserving features while smoothing
            from scipy.signal import savgol_filter
            window_length = min(21, len(noisy_data) // 10)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 5:
                smoothed = savgol_filter(noisy_data.values, window_length, 3)
                noisy_data[:] = smoothed
                
        elif smoothing_method == 'median':
            # Adaptive median filter
            window_size = 7
            smoothed = noisy_data.rolling(window=window_size, center=True, min_periods=1).median()
            noisy_data[:] = smoothed
            
        elif smoothing_method == 'kalman':
            # Simple Kalman-like filter using the stable eye as reference
            alpha = 0.7  # Trust factor for stable eye
            beta = 0.3   # Trust factor for noisy eye
            
            # Weighted combination biased toward stable eye for very noisy samples
            disparity = (noisy_data - stable_data).abs()
            high_disparity = disparity > disparity.quantile(0.9)
            
            smoothed = noisy_data.copy()
            smoothed.loc[high_disparity] = (alpha * stable_data.loc[high_disparity] + 
                                          beta * noisy_data.loc[high_disparity])
            noisy_data[:] = smoothed
        
        # Method 3: Replace extreme outliers with interpolated values
        if outlier_mask.sum() > 0:
            noisy_data.loc[outlier_mask] = np.nan
            noisy_data = noisy_data.interpolate(method='linear')
        
        # Put smoothed data back
        df.loc[valid_mask, noisy_col] = noisy_data
    
    # Recalculate noise metrics to show improvement
    if verbose:
        new_metrics = calculate_noise_metrics(f'x_{noisy_eye}', f'y_{noisy_eye}')
        original_score = left_metrics if noisy_eye == 'left' else right_metrics
        improvement = (1 - new_metrics['noise_level'] / original_score['noise_level']) * 100
        print(f"   ✓ Noise reduced by {improvement:.1f}%")
    
    return df


def enforce_gaze_quality_constraints(
    df: pd.DataFrame,
    max_gaze_disparity: float = 100,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Enforce gaze quality constraints: remove samples with excessive binocular disparity.
    For gaze data, we want both eyes looking at approximately the same screen location.

    Args:
        df: Input dataframe with gaze coordinates
        max_gaze_disparity: Maximum allowed gaze disparity in pixels (default 100px)
        verbose: Print constraint violations

    Returns:
        df: DataFrame with poor quality gaze samples removed
    """
    df = df.copy()

    # Find samples with excessive gaze disparity
    valid_mask = df['both_eyes_valid']
    
    if valid_mask.sum() == 0:
        if verbose:
            print("⚠️ No valid binocular samples to check")
        return df
    
    # Calculate gaze disparity for valid samples
    gaze_disparity = df.loc[valid_mask, 'disparity_total']
    
    # Find samples with excessive disparity
    excessive_disparity = valid_mask & (df['disparity_total'] > max_gaze_disparity)
    n_excessive = excessive_disparity.sum()
    
    if n_excessive > 0:
        if verbose:
            print(f"⚠️ Found {n_excessive} samples with excessive gaze disparity (>{max_gaze_disparity}px)")
            print(f"   ({n_excessive/valid_mask.sum()*100:.2f}% of valid data)")
            print("   → Removing poor quality gaze samples")

        # Invalidate excessive disparity samples
        df.loc[excessive_disparity, ['x_left', 'y_left', 'x_right', 'y_right']] = np.nan
        df.loc[excessive_disparity, 'both_eyes_valid'] = False

    # Update validity
    df['both_eyes_valid'] = (
        df['x_left'].notna() & df['x_right'].notna() &
        df['y_left'].notna() & df['y_right'].notna()
    )
    
    final_valid = df['both_eyes_valid'].sum()
    
    if verbose:
        # Show gaze disparity statistics
        remaining_disparity = df.loc[df['both_eyes_valid'], 'disparity_total']
        if len(remaining_disparity) > 0:
            print(f"\n📊 Gaze disparity statistics (remaining data):")
            print(f"   Mean: {remaining_disparity.mean():.1f}px")
            print(f"   Median: {remaining_disparity.median():.1f}px")
            print(f"   95th percentile: {remaining_disparity.quantile(0.95):.1f}px")
        
        print(f"\n✓ After gaze quality filtering: {final_valid} valid samples")

    return df


def detect_and_fix_eye_swap(
    df: pd.DataFrame,
    check_window: int = 1000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Detect if eye gaze labels were swapped during recording and fix if needed.
    For gaze data, we check if the "right eye" gaze is consistently to the left 
    of "left eye" gaze, which would indicate swapped labels.

    Args:
        df: Input dataframe
        check_window: Number of samples to check for swap detection
        verbose: Print swap detection info

    Returns:
        df: DataFrame with corrected eye labels
    """
    df = df.copy()

    # Check first valid window of data
    valid_samples = df[df['both_eyes_valid']].head(check_window)

    if len(valid_samples) < 100:
        if verbose:
            print("⚠️ Not enough valid samples to check for eye label swap")
        return df

    # For gaze data, check if there's a systematic bias in gaze positions
    # If eyes are consistently swapped, we might see the "right eye" gaze 
    # systematically to the left of "left eye" gaze
    x_diff = valid_samples['x_right'] - valid_samples['x_left']
    
    # Calculate how often right eye gaze is to the right of left eye gaze
    right_of_left_count = (x_diff > 0).sum()
    right_ratio = right_of_left_count / len(valid_samples)

    if verbose:
        print(f"✓ Gaze label check: {right_ratio*100:.1f}% of samples have right eye gaze to the right of left eye gaze")
        print(f"   Mean horizontal gaze difference: {x_diff.mean():.1f}px")

    # Only swap if there's a very strong systematic bias (>80% reversed)
    if right_ratio < 0.2:  # Less than 20% have correct orientation
        if verbose:
            print(f"🔄 Systematic gaze label swap detected!")
            print("   → Swapping eye labels throughout recording")

        # Swap all eye-related columns
        swap_columns = [
            ('x_left', 'x_right'),
            ('y_left', 'y_right'),
            ('pupil_left', 'pupil_right'),
            ('is_fixation_left', 'is_fixation_right'),
            ('is_saccade_left', 'is_saccade_right'),
            ('is_blink_left', 'is_blink_right'),
            ('velocity_left', 'velocity_right'),
        ]

        for left_col, right_col in swap_columns:
            if left_col in df.columns and right_col in df.columns:
                df[left_col], df[right_col] = df[right_col].copy(), df[left_col].copy()

        # Recalculate disparity
        df['disparity_x'] = df['x_left'] - df['x_right']
        df['disparity_y'] = df['y_left'] - df['y_right']
        df['disparity_total'] = np.sqrt(df['disparity_x']**2 + df['disparity_y']**2)

        if verbose:
            print("   ✓ Eye labels corrected")
    
    return df


def detect_and_handle_blinks(
    df: pd.DataFrame,
    gap_threshold: int = None,
    blink_window: int = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and handle blinks in eye-tracking data.
    Technical gaps (>threshold) are marked as NaN.
    Natural blinks (≤window) are interpolated.
    
    Args:
        df: Input dataframe
        gap_threshold: Samples to consider as gap (default from config)
        blink_window: Max samples for natural blink (default from config)
        verbose: Print blink statistics
        
    Returns:
        df: DataFrame with blinks handled
        blink_stats: Dictionary with blink statistics
    """
    df = df.copy()
    
    # Use defaults from config
    if gap_threshold is None:
        gap_threshold = PREPROCESSING_PARAMS['blink_gap_threshold']
    if blink_window is None:
        blink_window = PREPROCESSING_PARAMS['blink_window']
    
    # Initialize blink columns if not present
    for col in ['is_blink_left', 'is_blink_right']:
        if col not in df.columns:
            df[col] = False
    
    # Function to find consecutive missing periods
    def find_missing_periods(missing_mask):
        periods = []
        in_period = False
        start_idx = 0
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and not in_period:
                start_idx = i
                in_period = True
            elif not is_missing and in_period:
                periods.append((start_idx, i - 1))
                in_period = False
        
        if in_period:
            periods.append((start_idx, len(missing_mask) - 1))
        
        return periods
    
    # Process each eye
    blink_stats = {}
    
    for eye in ['left', 'right']:
        missing = df[f'x_{eye}'].isna() | df[f'y_{eye}'].isna()
        periods = find_missing_periods(missing)
        
        technical_gaps = 0
        natural_blinks = 0
        interpolated_samples = 0
        
        for start, end in periods:
            duration = end - start + 1
            
            if gap_threshold < duration <= blink_window:
                # Natural blink - interpolate
                if start > 0 and end < len(df) - 1:
                    # Linear interpolation
                    for col in [f'x_{eye}', f'y_{eye}']:
                        df.loc[start:end, col] = np.interp(
                            np.arange(start, end + 1),
                            [start - 1, end + 1],
                            [df.loc[start - 1, col], df.loc[end + 1, col]]
                        )
                    # Mark as blink
                    df.loc[start:end, f'is_blink_{eye}'] = True
                    natural_blinks += 1
                    interpolated_samples += duration
            elif duration > blink_window:
                # Technical gap
                technical_gaps += 1
        
        blink_stats[f'{eye}_eye'] = {
            'technical_gaps': technical_gaps,
            'natural_blinks': natural_blinks,
            'interpolated_samples': interpolated_samples
        }
    
    # Update validity
    df['both_eyes_valid'] = (
        df['x_left'].notna() & df['x_right'].notna() &
        df['y_left'].notna() & df['y_right'].notna()
    )
    
    if verbose:
        print("👁️ Blink Detection and Handling")
        print(f"  Left eye: {blink_stats['left_eye']['natural_blinks']} blinks, "
              f"{blink_stats['left_eye']['technical_gaps']} gaps")
        print(f"  Right eye: {blink_stats['right_eye']['natural_blinks']} blinks, "
              f"{blink_stats['right_eye']['technical_gaps']} gaps")
    
    return df, blink_stats


def preprocess_pupil_size(
    df: pd.DataFrame,
    min_pupil: float = None,
    max_pupil: float = None,
    window_size: int = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess pupil size data by removing implausible values and smoothing.
    
    Args:
        df: Input dataframe
        min_pupil: Minimum valid pupil size (default from config)
        max_pupil: Maximum valid pupil size (default from config)
        window_size: Gaussian smoothing window (default from config)
        verbose: Print pupil statistics
        
    Returns:
        df: DataFrame with preprocessed pupil data
        pupil_stats: Dictionary with pupil statistics
    """
    df = df.copy()
    
    # Use defaults from config
    if min_pupil is None:
        min_pupil = PREPROCESSING_PARAMS['pupil_min']
    if max_pupil is None:
        max_pupil = PREPROCESSING_PARAMS['pupil_max']
    if window_size is None:
        window_size = PREPROCESSING_PARAMS['pupil_smoothing_window']
    
    adaptation_samples = int(PREPROCESSING_PARAMS['pupil_adaptation_time'] * SAMPLING_RATE)
    
    pupil_stats = {}
    
    for eye in ['left', 'right']:
        col = f'pupil_{eye}'
        if col not in df.columns:
            continue
        
        original_data = df[col].copy()
        
        # Count implausible values
        zero_pupils = (original_data == 0).sum()
        too_small = ((original_data > 0) & (original_data < min_pupil)).sum()
        too_large = (original_data > max_pupil).sum()
        
        # Remove implausible values
        df.loc[(df[col] == 0) | (df[col] < min_pupil) | (df[col] > max_pupil), col] = np.nan
        
        # Exclude adaptation period
        if len(df) > adaptation_samples:
            df.loc[:adaptation_samples-1, col] = np.nan
        
        # Detect and remove sharp changes
        if df[col].notna().sum() > 10:
            pupil_diff = df[col].diff().abs()
            threshold = pupil_diff.quantile(0.99)
            sharp_changes = pupil_diff > threshold
            sharp_change_count = sharp_changes.sum()
            df.loc[sharp_changes, col] = np.nan
        else:
            sharp_change_count = 0
        
        # Apply Gaussian smoothing
        valid_mask = df[col].notna()
        if valid_mask.sum() > window_size:
            valid_indices = np.where(valid_mask)[0]
            valid_values = df.loc[valid_mask, col].values
            
            # Apply smoothing
            sigma = window_size / 4
            smoothed_values = gaussian_filter1d(valid_values, sigma=sigma, mode='nearest')
            
            # Put smoothed values back
            df.loc[valid_indices, col] = smoothed_values
        
        # Calculate statistics
        final_valid = df[col].notna().sum()
        original_valid = original_data.notna().sum()
        
        pupil_stats[f'{eye}_eye'] = {
            'original_valid': original_valid,
            'zero_pupils': zero_pupils,
            'too_small': too_small,
            'too_large': too_large,
            'sharp_changes': sharp_change_count,
            'final_valid': final_valid,
            'removed_total': original_valid - final_valid,
            'removal_percentage': (original_valid - final_valid) / original_valid * 100 if original_valid > 0 else 0
        }
    
    if verbose:
        print("👁️ Pupil Size Preprocessing")
        for eye in ['left', 'right']:
            if f'{eye}_eye' in pupil_stats:
                stats = pupil_stats[f'{eye}_eye']
                print(f"  {eye.capitalize()} eye: {stats['removal_percentage']:.1f}% removed "
                      f"({stats['final_valid']:,} valid samples)")
    
    return df, pupil_stats


def preprocess_eye_tracking_data(
    file_path: str,
    verbose: bool = True,
    create_plots: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete eye-tracking data preprocessing pipeline.
    
    Args:
        file_path: Path to eye-tracking CSV file
        verbose: Print progress and statistics
        create_plots: Generate visualization plots
        
    Returns:
        df: Preprocessed DataFrame
        preprocessing_info: Dictionary with preprocessing statistics
    """
    if verbose:
        print(f"\n🚀 Complete Preprocessing Pipeline")
        print(f"File: {os.path.basename(file_path)}")
        print("="*60)

    # Step 1: Load data
    df = pd.read_csv(file_path)
    initial_samples = len(df)
    subject_id = os.path.basename(file_path).split('_')[0]

    # Save original for comparison
    df_original = df.copy()

    # Remove empty columns (inline, simplified)
    missing_proportion = df.isnull().sum() / len(df)
    cols_to_remove = missing_proportion[missing_proportion > PREPROCESSING_PARAMS['missing_data_threshold']].index.tolist()
    if cols_to_remove and verbose:
        print(f"Removing {len(cols_to_remove)} columns with >{PREPROCESSING_PARAMS['missing_data_threshold']*100:.0f}% missing data:")
        for col in cols_to_remove:
            print(f"  - {col}: {missing_proportion[col]*100:.1f}% missing")
    df = df.drop(columns=cols_to_remove)

    if verbose:
        print(f"✓ Step 1: Loaded data ({initial_samples:,} samples)")

    # Step 2: Add helper features (inline, optimized)
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
    df['velocity_left'] = np.sqrt(df['x_left'].diff()**2 + df['y_left'].diff()**2) * SAMPLING_RATE
    df['velocity_right'] = np.sqrt(df['x_right'].diff()**2 + df['y_right'].diff()**2) * SAMPLING_RATE
    df['disparity_x'] = df['x_left'] - df['x_right']
    df['disparity_y'] = df['y_left'] - df['y_right']
    df['disparity_total'] = np.sqrt(df['disparity_x']**2 + df['disparity_y']**2)
    df['both_eyes_valid'] = (df['x_left'].notna() & df['x_right'].notna() & df['y_left'].notna() & df['y_right'].notna())
    
    # Also add to original for plotting
    df_original = add_helper_features(df_original)
    if verbose:
        print("✓ Step 2: Added helper features")

    # Save intermediate stages for visualization
    pipeline_stages = []
    pipeline_titles = []

    # Step 3: Detect and fix eye swap if needed
    df = detect_and_fix_eye_swap(df, verbose=verbose)
    if verbose:
        print("✓ Step 3: Checked gaze labeling")
    
    # Save before stabilization for comparison
    df_before_stabilization = df.copy()
    
    # Step 4: Detect and stabilize noisy eye
    df = detect_and_stabilize_noisy_eye(df, verbose=verbose)
    if verbose:
        print("✓ Step 4: Stabilized noisy eye tracking")
    pipeline_stages.append(df.copy())
    pipeline_titles.append("After Noise Stabilization")

    # Step 5: Detect and handle blinks
    df, blink_stats = detect_and_handle_blinks(df, verbose=verbose)
    after_blinks = len(df)
    pipeline_stages.append(df.copy())
    pipeline_titles.append("After Blink Handling")

    # Step 6: Enforce binocular validity
    df = enforce_binocular_validity(df)
    binocular_samples = df['both_eyes_valid'].sum()
    pipeline_stages.append(df.copy())
    pipeline_titles.append("After Binocular Enforcement")

    # Step 7: Apply calibration correction
    df, correction_info = automatic_binocular_calibration(df, verbose=verbose)
    after_calibration = len(df)
    pipeline_stages.append(df.copy())
    pipeline_titles.append("After Calibration")
    
    # Step 8: Enforce gaze quality constraints
    df = enforce_gaze_quality_constraints(df, verbose=verbose)
    after_gaze_quality = df['both_eyes_valid'].sum()
    pipeline_stages.append(df.copy())
    pipeline_titles.append("After Gaze Quality Filtering")

    # Step 9: Remove extreme disparities
    df = remove_extreme_disparities(df)
    valid_after_disparity = df['both_eyes_valid'].sum()
    pipeline_stages.append(df.copy())
    pipeline_titles.append("After Disparity Filtering")

    # Step 10: Round coordinates to pixels (inline)
    for col in ['x_left', 'y_left', 'x_right', 'y_right']:
        if col in df.columns:
            df[col] = df[col].round().astype('Int64')
    if verbose:
        print("🎯 Coordinates rounded to integer pixels")

    # Step 11: Preprocess pupil size
    df, pupil_stats = preprocess_pupil_size(df, verbose=verbose)

    # Calculate final statistics
    final_valid_samples = df['both_eyes_valid'].sum()

    preprocessing_info = {
        'subject_id': subject_id,
        'file': os.path.basename(file_path),
        'initial_samples': initial_samples,
        'final_valid_samples': final_valid_samples,
        'retention_rate': final_valid_samples / initial_samples * 100,
        'blink_stats': blink_stats,
        'calibration_correction': correction_info,
        'pupil_stats': pupil_stats,
        'steps': {
            'after_blinks': after_blinks,
            'after_binocular': binocular_samples,
            'after_calibration': after_calibration,
            'after_gaze_quality': after_gaze_quality,
            'after_disparity': valid_after_disparity,
            'final': final_valid_samples
        }
    }

    if verbose:
        print("\n" + "="*60)
        print(f"📊 Final Summary:")
        print(f"   Initial samples: {initial_samples:,}")
        print(f"   Final valid samples: {final_valid_samples:,}")
        print(f"   Data retention: {preprocessing_info['retention_rate']:.1f}%")
        print("="*60)

    return df, preprocessing_info