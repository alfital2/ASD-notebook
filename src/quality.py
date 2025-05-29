"""
Quality assessment functions for eye-tracking data.
"""

import numpy as np
import pandas as pd
from typing import Dict

# Configuration
SAMPLING_RATE = 500  # Hz

# Analysis parameters
ANALYSIS_PARAMS = {
    'min_valid_data': 0.60,        # minimum 60% valid data required
    'window_size': 500,            # 1-second windows for temporal analysis
}


def calculate_alignment_quality(df: pd.DataFrame) -> pd.Series:
    """
    Evaluate how well left/right eyes track together.
    
    Args:
        df: DataFrame with disparity data
        
    Returns:
        alignment_scores: Quality scores (0-1 scale)
    """
    if 'disparity_total' not in df.columns:
        return pd.Series(0.5, index=df.index)  # Neutral score
    
    disparity = df['disparity_total']
    
    # Quality categories with weighted scores
    excellent = (disparity < 10).astype(int) * 1.0    # <10px = perfect
    good = ((disparity >= 10) & (disparity < 30)).astype(int) * 0.8  # 10-30px = good
    fair = ((disparity >= 30) & (disparity < 50)).astype(int) * 0.5  # 30-50px = fair
    poor = (disparity >= 50).astype(int) * 0.2       # >50px = poor
    
    return excellent + good + fair + poor


def calculate_plausibility_score(df: pd.DataFrame) -> pd.Series:
    """
    Check if eye movements are physiologically plausible.
    
    Args:
        df: DataFrame with velocity and pupil data
        
    Returns:
        plausibility_scores: Quality scores (0-1 scale)
    """
    # Physiological velocity limit (humans can't move eyes >1000°/s)
    max_velocity = 1000  # pixels/second at typical viewing distance
    
    # Use existing velocity columns or calculate if missing
    if 'velocity_left' in df.columns:
        velocity_left = df['velocity_left'] 
    else:
        velocity_left = np.sqrt(df['x_left'].diff()**2 + df['y_left'].diff()**2) * SAMPLING_RATE
        
    if 'velocity_right' in df.columns:
        velocity_right = df['velocity_right']
    else:
        velocity_right = np.sqrt(df['x_right'].diff()**2 + df['y_right'].diff()**2) * SAMPLING_RATE
    
    # Check if velocities are physiologically plausible
    velocity_ok_left = (velocity_left < max_velocity).astype(float)
    velocity_ok_right = (velocity_right < max_velocity).astype(float)
    
    # Check pupil size consistency (if available)
    if 'pupil_left' in df.columns and df['pupil_left'].notna().sum() > 100:
        pupil_stable = (df['pupil_left'].rolling(10, min_periods=1).std() < 100).astype(float)
        pupil_stable = pupil_stable.fillna(0.5)  # neutral score for missing data
    else:
        pupil_stable = pd.Series(0.5, index=df.index)  # neutral score when no pupil data
    
    # Combine metrics (velocity is more important than pupil stability)
    plausibility = (velocity_ok_left * 0.4 + velocity_ok_right * 0.4 + pupil_stable * 0.2)
    return plausibility.fillna(0)


def calculate_snr_score(df: pd.DataFrame, window_size: int = 100) -> pd.Series:
    """
    Calculate signal-to-noise ratio based on movement characteristics.
    
    Args:
        df: DataFrame with coordinate data
        window_size: Rolling window size for calculations
        
    Returns:
        snr_scores: Quality scores (0-1 scale)
    """
    # Calculate local noise (high-frequency jitter) for both eyes
    x_noise_left = df['x_left'].diff().diff().abs().rolling(window_size, min_periods=10).mean()
    x_noise_right = df['x_right'].diff().diff().abs().rolling(window_size, min_periods=10).mean()
    
    # Calculate signal strength (meaningful movements)
    x_signal_left = df['x_left'].rolling(window_size, min_periods=10).std()
    x_signal_right = df['x_right'].rolling(window_size, min_periods=10).std()
    
    # Calculate SNR for both eyes (higher = better quality)
    snr_left = x_signal_left / (x_noise_left + 1)  # +1 to avoid division by zero
    snr_right = x_signal_right / (x_noise_right + 1)
    
    # Average the two eyes and normalize to 0-1 scale
    snr_combined = (snr_left + snr_right) / 2
    
    # Normalize to 0-1 range (typical SNR values range from 0-50)
    snr_normalized = np.clip(snr_combined / 50, 0, 1)
    
    return snr_normalized.fillna(0)


def calculate_comprehensive_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all quality metrics for comprehensive assessment.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        quality_metrics: DataFrame with all quality dimensions
    """
    print("🔍 Calculating comprehensive quality metrics...")
    
    # Basic validity (existing)
    basic_validity = df['both_eyes_valid'].astype(float)
    
    # Enhanced quality metrics
    alignment_quality = calculate_alignment_quality(df)
    plausibility_score = calculate_plausibility_score(df)
    snr_score = calculate_snr_score(df)
    
    # Create quality metrics dataframe
    quality_metrics = pd.DataFrame({
        'time_seconds': df['time_seconds'],
        'basic_validity': basic_validity,
        'alignment_quality': alignment_quality,
        'plausibility_score': plausibility_score,
        'snr_score': snr_score
    })
    
    # Calculate composite quality score (weighted average)
    weights = {
        'basic_validity': 0.4,      # Most important: data exists
        'alignment_quality': 0.3,   # Second: eyes track together
        'plausibility_score': 0.2,  # Third: movements are human-like
        'snr_score': 0.1           # Fourth: signal is clean
    }
    
    quality_metrics['composite_quality'] = (
        quality_metrics['basic_validity'] * weights['basic_validity'] +
        quality_metrics['alignment_quality'] * weights['alignment_quality'] +
        quality_metrics['plausibility_score'] * weights['plausibility_score'] +
        quality_metrics['snr_score'] * weights['snr_score']
    )
    
    return quality_metrics


def calculate_quality_trends(
    quality_metrics: pd.DataFrame, 
    window_size: int = None
) -> Dict:
    """
    Calculate quality trends in sliding windows.
    
    Args:
        quality_metrics: DataFrame with quality metrics
        window_size: Sliding window size (default from ANALYSIS_PARAMS)
        
    Returns:
        trends_data: Dictionary with trend values
    """
    if window_size is None:
        window_size = ANALYSIS_PARAMS['window_size']
    
    # Calculate sliding window averages
    step_size = window_size // 4  # Overlap windows by 75%
    time_points = []
    basic_validity_vals = []
    alignment_quality_vals = []
    plausibility_vals = []
    snr_vals = []
    composite_vals = []
    
    for i in range(0, len(quality_metrics) - window_size, step_size):
        window = quality_metrics.iloc[i:i+window_size]
        
        # Time point (middle of window)
        time_points.append(window['time_seconds'].iloc[window_size//2])
        
        # Calculate averages for each metric
        basic_validity_vals.append(window['basic_validity'].mean() * 100)
        alignment_quality_vals.append(window['alignment_quality'].mean() * 100)
        plausibility_vals.append(window['plausibility_score'].mean() * 100)
        snr_vals.append(window['snr_score'].mean() * 100)
        composite_vals.append(window['composite_quality'].mean() * 100)
    
    # Create trends dictionary
    trends_data = {
        'time_points': time_points,
        'basic_validity': basic_validity_vals,
        'alignment_quality': alignment_quality_vals,
        'plausibility': plausibility_vals,
        'snr': snr_vals,
        'composite': composite_vals
    }
    
    # Print summary statistics
    print("\n📈 Quality Metrics Summary:")
    print("="*60)
    print(f"{'Metric':<25} {'Mean':<8} {'Min':<8} {'Max':<8} {'Std':<8}")
    print("-"*60)
    print(f"{'Basic Validity':<25} {np.mean(basic_validity_vals):<8.1f} {np.min(basic_validity_vals):<8.1f} {np.max(basic_validity_vals):<8.1f} {np.std(basic_validity_vals):<8.1f}")
    print(f"{'Binocular Alignment':<25} {np.mean(alignment_quality_vals):<8.1f} {np.min(alignment_quality_vals):<8.1f} {np.max(alignment_quality_vals):<8.1f} {np.std(alignment_quality_vals):<8.1f}")
    print(f"{'Physiological Plausibility':<25} {np.mean(plausibility_vals):<8.1f} {np.min(plausibility_vals):<8.1f} {np.max(plausibility_vals):<8.1f} {np.std(plausibility_vals):<8.1f}")
    print(f"{'Signal-to-Noise Ratio':<25} {np.mean(snr_vals):<8.1f} {np.min(snr_vals):<8.1f} {np.max(snr_vals):<8.1f} {np.std(snr_vals):<8.1f}")
    print(f"{'Composite Quality':<25} {np.mean(composite_vals):<8.1f} {np.min(composite_vals):<8.1f} {np.max(composite_vals):<8.1f} {np.std(composite_vals):<8.1f}")
    
    return trends_data


def assess_data_quality(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Perform a comprehensive assessment of eye-tracking data quality.
    
    Args:
        df: Preprocessed DataFrame
        verbose: Print quality assessment info
        
    Returns:
        quality_assessment: Dictionary with quality metrics
    """
    # Calculate quality metrics
    quality_metrics = calculate_comprehensive_quality_metrics(df)
    
    # Calculate quality trends
    quality_trends = calculate_quality_trends(quality_metrics)
    
    # Overall quality metrics
    overall_metrics = {
        'valid_samples_ratio': df['both_eyes_valid'].mean(),
        'basic_validity_mean': quality_metrics['basic_validity'].mean(),
        'alignment_quality_mean': quality_metrics['alignment_quality'].mean(),
        'plausibility_score_mean': quality_metrics['plausibility_score'].mean(),
        'snr_score_mean': quality_metrics['snr_score'].mean(),
        'composite_quality_mean': quality_metrics['composite_quality'].mean(),
    }
    
    # Create quality assessment dictionary
    quality_assessment = {
        'quality_metrics': quality_metrics,
        'quality_trends': quality_trends,
        'overall_metrics': overall_metrics,
    }
    
    # Print overall quality assessment
    if verbose:
        print("\n🎯 Overall Quality Assessment:")
        print("="*60)
        print(f"Basic Validity: {overall_metrics['basic_validity_mean']*100:.1f}%")
        print(f"Binocular Alignment: {overall_metrics['alignment_quality_mean']*100:.1f}%")
        print(f"Physiological Plausibility: {overall_metrics['plausibility_score_mean']*100:.1f}%")
        print(f"Signal-to-Noise Ratio: {overall_metrics['snr_score_mean']*100:.1f}%")
        print(f"Composite Quality: {overall_metrics['composite_quality_mean']*100:.1f}%")
        
        # Quality rating
        composite_score = overall_metrics['composite_quality_mean'] * 100
        rating = 'Excellent' if composite_score >= 90 else 'Good' if composite_score >= 80 else 'Acceptable' if composite_score >= 70 else 'Poor'
        print(f"\nOverall Rating: {rating} ({composite_score:.1f}%)")
    
    return quality_assessment