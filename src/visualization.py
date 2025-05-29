"""
Visualization functions for eye-tracking data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from typing import Dict, Tuple, List, Optional, Union

# Configuration
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1024
SAMPLING_RATE = 500  # Hz

# Visualization settings
VIZ_PARAMS = {
    'color_left': 'blue',
    'color_right': 'red',
    'alpha_scatter': 0.3,
    'alpha_line': 0.7,
    'figure_size': (12, 8),
    'grid_alpha': 0.3,
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    'plot_sample_rate': 10,        # sample every N points for scatter plots
    'figure_dpi': 100,             # figure resolution
}


def plot_before_after(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature: str,
    title: str = None,
    figsize: Tuple[int, int] = None
) -> plt.Figure:
    """
    Generic before/after comparison plot for any feature.
    
    Args:
        df_before: Original dataframe
        df_after: Processed dataframe
        feature: Feature column to plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    if figsize is None:
        figsize = VIZ_PARAMS['figure_size']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Before plot
    if feature in df_before.columns:
        ax1.plot(df_before['time_seconds'], df_before[feature],
                 alpha=VIZ_PARAMS['alpha_line'], linewidth=0.5)
        ax1.set_title(f'Before: {feature}')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel(feature)
        ax1.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
    
    # After plot
    if feature in df_after.columns:
        ax2.plot(df_after['time_seconds'], df_after[feature],
                 alpha=VIZ_PARAMS['alpha_line'], linewidth=0.5)
        ax2.set_title(f'After: {feature}')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel(feature)
        ax2.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_gaze_trajectory(
    df: pd.DataFrame,
    eye: str = 'both',
    sample_rate: int = None,
    ax: plt.Axes = None,
    title: str = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot gaze trajectory for one or both eyes.
    
    Args:
        df: DataFrame with eye tracking data
        eye: 'left', 'right', or 'both'
        sample_rate: Sample every N points (default from config)
        ax: Existing axis to plot on
        title: Plot title
        
    Returns:
        fig or ax: Figure if ax is None, otherwise axis
    """
    if sample_rate is None:
        sample_rate = PREPROCESSING_PARAMS['plot_sample_rate']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=VIZ_PARAMS['figure_size'])
        return_fig = True
    else:
        return_fig = False
    
    # Sample data to avoid overplotting
    sampled_df = df[::sample_rate]
    
    # Plot eye data
    if eye in ['left', 'both']:
        valid = sampled_df.dropna(subset=['x_left', 'y_left'])
        ax.scatter(valid['x_left'], valid['y_left'],
                   c=VIZ_PARAMS['color_left'], alpha=VIZ_PARAMS['alpha_scatter'],
                   s=1, label='Left eye')
    
    if eye in ['right', 'both']:
        valid = sampled_df.dropna(subset=['x_right', 'y_right'])
        ax.scatter(valid['x_right'], valid['y_right'],
                   c=VIZ_PARAMS['color_right'], alpha=VIZ_PARAMS['alpha_scatter'],
                   s=1, label='Right eye')
    
    # Add screen boundaries
    screen_rect = Rectangle((0, 0), SCREEN_WIDTH, SCREEN_HEIGHT,
                            fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(screen_rect)
    
    # Set limits and labels
    margin = 100
    ax.set_xlim(-margin, SCREEN_WIDTH + margin)
    ax.set_ylim(-margin, SCREEN_HEIGHT + margin)
    ax.invert_yaxis()  # Invert Y axis for screen coordinates
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('Y coordinate (pixels)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Gaze Trajectory ({eye.capitalize()} eye)')
    
    ax.legend()
    ax.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
    ax.set_aspect('equal', adjustable='box')
    
    return fig if return_fig else ax


def plot_disparity_analysis(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Comprehensive binocular disparity analysis plot.
    
    Args:
        df: DataFrame with disparity data
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Disparity over time
    ax = axes[0, 0]
    ax.plot(df['time_seconds'], df['disparity_total'],
            color='purple', alpha=0.7, linewidth=0.5)
    ax.axhline(y=10, color='green', linestyle='--', label='Good (<10 px)')
    ax.axhline(y=30, color='orange', linestyle='--', label='Acceptable (<30 px)')
    ax.axhline(y=50, color='red', linestyle='--', label='Poor (>50 px)')
    ax.set_ylabel('Binocular Disparity (pixels)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Disparity Throughout Recording')
    ax.set_ylim(0, min(200, df['disparity_total'].quantile(0.99) * 1.1))
    ax.legend()
    ax.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
    
    # 2. Disparity distribution
    ax = axes[0, 1]
    valid_disparity = df['disparity_total'].dropna()
    ax.hist(valid_disparity, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=valid_disparity.median(), color='red', linestyle='--',
               label=f'Median: {valid_disparity.median():.1f} px')
    ax.axvline(x=valid_disparity.mean(), color='blue', linestyle='--',
               label=f'Mean: {valid_disparity.mean():.1f} px')
    ax.set_xlabel('Disparity (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Disparity Distribution')
    ax.legend()
    ax.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
    
    # 3. Disparity pattern (X vs Y) with TIME COLORBAR and ABSOLUTE VALUES
    ax = axes[1, 0]
    sample_data = df[::PREPROCESSING_PARAMS['plot_sample_rate']].dropna(
        subset=['disparity_x', 'disparity_y'])
    
    if len(sample_data) > 0:
        # Use absolute values for disparities
        abs_disparity_x = sample_data['disparity_x'].abs()
        abs_disparity_y = sample_data['disparity_y'].abs()
        
        scatter = ax.scatter(abs_disparity_x, abs_disparity_y,
                             alpha=0.6, s=8, c=sample_data['time_seconds'], 
                             cmap='viridis', edgecolors='none')
        
        # Add colorbar for time reference
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Time (seconds)', rotation=270, labelpad=15)
        
        # Add reference circles (now from origin since all values are positive)
        for radius, color in [(10, 'green'), (30, 'orange'), (50, 'red')]:
            circle = Circle((0, 0), radius, fill=False, color=color,
                            linestyle='--', alpha=0.7, linewidth=2)
            ax.add_patch(circle)
        
        ax.set_xlabel('Absolute Horizontal Disparity (pixels)')
        ax.set_ylabel('Absolute Vertical Disparity (pixels)')
        ax.set_title('Binocular Disparity Pattern (absolute values, colored by time)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
        
        # Set limits to start from 0
        ax.set_xlim(0, max(abs_disparity_x.max() * 1.1, 100))
        ax.set_ylim(0, max(abs_disparity_y.max() * 1.1, 100))
    
    # 4. Validity over time
    ax = axes[1, 1]
    from src.quality import ANALYSIS_PARAMS
    window_size = ANALYSIS_PARAMS['window_size']
    validity_over_time = []
    time_points = []
    
    for i in range(0, len(df) - window_size, window_size // 2):
        window = df.iloc[i:i+window_size]
        validity_percent = window['both_eyes_valid'].sum() / len(window) * 100
        validity_over_time.append(validity_percent)
        time_points.append(window['time_seconds'].mean())
    
    ax.plot(time_points, validity_over_time, 'g-', linewidth=2)
    ax.axhline(y=60, color='orange', linestyle='--', label='60% threshold')
    ax.axhline(y=80, color='green', linestyle='--', label='80% threshold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Data Validity (%)')
    ax.set_title('Data Validity Over Time (1-second windows)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=VIZ_PARAMS['grid_alpha'])
    
    plt.suptitle('Binocular Eye-Tracking Quality Analysis', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_preprocessing_summary(
    preprocessing_info: Dict,
    save_path: str = None
) -> plt.Figure:
    """
    Create a summary plot of preprocessing statistics.
    
    Args:
        preprocessing_info: Dictionary with preprocessing statistics
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data retention through pipeline
    steps = ['Initial', 'Blinks', 'Binocular', 'Calibration', 'Disparity', 'Final']
    values = [
        preprocessing_info['initial_samples'],
        preprocessing_info['steps'].get('after_blinks', preprocessing_info['initial_samples']),
        preprocessing_info['steps']['after_binocular'],
        preprocessing_info['steps']['after_calibration'],
        preprocessing_info['steps']['after_disparity'],
        preprocessing_info['final_valid_samples']
    ]
    
    ax1.plot(steps, values, 'o-', linewidth=2, markersize=8)
    ax1.set_ylabel('Valid Samples')
    ax1.set_title('Data Retention Through Pipeline')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (step, value) in enumerate(zip(steps, values)):
        percent = value / preprocessing_info['initial_samples'] * 100
        ax1.text(i, value + 500, f'{percent:.1f}%', ha='center', va='bottom')
    
    # Preprocessing impact breakdown
    categories = ['Blinks\nInterpolated', 'Monocular\nRemoved', 'Extreme\nDisparity', 'Other']
    
    blink_impact = (preprocessing_info['blink_stats']['left_eye']['interpolated_samples'] +
                    preprocessing_info['blink_stats']['right_eye']['interpolated_samples']) / 2
    
    impacts = [
        blink_impact,
        preprocessing_info['initial_samples'] - preprocessing_info['steps']['after_binocular'],
        preprocessing_info['steps']['after_calibration'] - preprocessing_info['steps']['after_disparity'],
        preprocessing_info['initial_samples'] - preprocessing_info['final_valid_samples'] - 
        (preprocessing_info['initial_samples'] - preprocessing_info['steps']['after_binocular']) -
        (preprocessing_info['steps']['after_calibration'] - preprocessing_info['steps']['after_disparity'])
    ]
    
    # Ensure no negative values
    impacts = [max(0, impact) for impact in impacts]
    
    ax2.bar(categories, impacts, color=['green', 'orange', 'red', 'gray'])
    ax2.set_ylabel('Samples Affected')
    ax2.set_title('Preprocessing Impact by Category')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    retention_rate = preprocessing_info['retention_rate']
    plt.suptitle(f'Preprocessing Summary - {retention_rate:.1f}% Data Retained', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PREPROCESSING_PARAMS['figure_dpi'], bbox_inches='tight')
    
    return fig


def plot_pupil_analysis(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    figsize: Tuple[int, int] = (18, 14)
) -> plt.Figure:
    """
    Comprehensive pupil size analysis showing before/after preprocessing.
    
    Args:
        df_before: Original dataframe
        df_after: Preprocessed dataframe
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Row 1: Full time series
    ax = axes[0, 0]
    ax.plot(df_before['time_seconds'], df_before['pupil_left'], 
            color='blue', alpha=0.5, linewidth=0.5, label='Left pupil')
    ax.plot(df_before['time_seconds'], df_before['pupil_right'], 
            color='red', alpha=0.5, linewidth=0.5, label='Right pupil')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pupil Size (arbitrary units)')
    ax.set_title('BEFORE: Raw Pupil Size Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    valid_left = df_after[df_after['pupil_left'].notna()]
    valid_right = df_after[df_after['pupil_right'].notna()]
    ax.plot(valid_left['time_seconds'], valid_left['pupil_left'], 
            color='blue', alpha=0.7, linewidth=1, label='Left pupil')
    ax.plot(valid_right['time_seconds'], valid_right['pupil_right'], 
            color='red', alpha=0.7, linewidth=1, label='Right pupil')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pupil Size (arbitrary units)')
    ax.set_title('AFTER: Preprocessed Pupil Size (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: Zoomed view (10-20 seconds)
    ax = axes[1, 0]
    mask = (df_before['time_seconds'] >= 10) & (df_before['time_seconds'] <= 20)
    ax.plot(df_before.loc[mask, 'time_seconds'], df_before.loc[mask, 'pupil_left'], 
            'b-', alpha=0.7, linewidth=1, marker='o', markersize=2, label='Left pupil')
    ax.plot(df_before.loc[mask, 'time_seconds'], df_before.loc[mask, 'pupil_right'], 
            'r-', alpha=0.7, linewidth=1, marker='o', markersize=2, label='Right pupil')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pupil Size')
    ax.set_title('BEFORE: Raw Data (10-20s zoom)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    mask = (df_after['time_seconds'] >= 10) & (df_after['time_seconds'] <= 20)
    valid_left = df_after[mask & df_after['pupil_left'].notna()]
    valid_right = df_after[mask & df_after['pupil_right'].notna()]
    ax.plot(valid_left['time_seconds'], valid_left['pupil_left'], 
            'b-', alpha=0.7, linewidth=2, label='Left pupil')
    ax.plot(valid_right['time_seconds'], valid_right['pupil_right'], 
            'r-', alpha=0.7, linewidth=2, label='Right pupil')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pupil Size')
    ax.set_title('AFTER: Smoothed Data (10-20s zoom)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 3: Distributions
    ax = axes[2, 0]
    left_before = df_before['pupil_left'].dropna()
    right_before = df_before['pupil_right'].dropna()
    ax.hist(left_before, bins=50, alpha=0.5, color='blue', 
            label=f'Left (n={len(left_before)})', density=True)
    ax.hist(right_before, bins=50, alpha=0.5, color='red', 
            label=f'Right (n={len(right_before)})', density=True)
    ax.set_xlabel('Pupil Size')
    ax.set_ylabel('Density')
    ax.set_title('BEFORE: Pupil Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    left_after = df_after['pupil_left'].dropna()
    right_after = df_after['pupil_right'].dropna()
    ax.hist(left_after, bins=50, alpha=0.5, color='blue', 
            label=f'Left (n={len(left_after)})', density=True)
    ax.hist(right_after, bins=50, alpha=0.5, color='red', 
            label=f'Right (n={len(right_after)})', density=True)
    ax.set_xlabel('Pupil Size')
    ax.set_ylabel('Density')
    ax.set_title('AFTER: Pupil Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Pupil Size Data: Before vs After Preprocessing', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_preprocessing_pipeline(
    df_list: List[pd.DataFrame],
    titles: List[str],
    figsize: Tuple[int, int] = (20, 16)
) -> plt.Figure:
    """
    Visualize data through each preprocessing step.
    
    Args:
        df_list: List of dataframes at each pipeline stage
        titles: List of titles for each stage
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    n_stages = len(df_list)
    fig, axes = plt.subplots(n_stages, 2, figsize=figsize)
    
    if n_stages == 1:
        axes = axes.reshape(1, -1)
    
    for i, (df, title) in enumerate(zip(df_list, titles)):
        # Left: Gaze trajectory
        ax = axes[i, 0]
        valid = df[::10].dropna(subset=['x_left', 'y_left', 'x_right', 'y_right'])
        ax.scatter(valid['x_left'], valid['y_left'], c='blue', alpha=0.3, s=1, label='Left')
        ax.scatter(valid['x_right'], valid['y_right'], c='red', alpha=0.3, s=1, label='Right')
        ax.set_xlim(-200, SCREEN_WIDTH + 200)
        ax.set_ylim(-200, SCREEN_HEIGHT + 200)
        ax.invert_yaxis()
        ax.set_title(f'{title} - Gaze Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: Validity over time
        ax = axes[i, 1]
        from src.quality import ANALYSIS_PARAMS
        window_size = ANALYSIS_PARAMS['window_size']
        validity = []
        time_points = []
        
        for j in range(0, len(df) - window_size, window_size // 2):
            window = df.iloc[j:j+window_size]
            valid_pct = window['both_eyes_valid'].sum() / len(window) * 100
            validity.append(valid_pct)
            time_points.append(window['time_seconds'].mean())
        
        ax.plot(time_points, validity, 'g-', linewidth=2)
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Data Validity (%)')
        ax.set_title(f'{title} - Validity Over Time')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Eye-Tracking Data Through Preprocessing Pipeline', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_blink_analysis(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Analyze and visualize blink patterns.
    
    Args:
        df: Dataframe with blink data
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Blink timeline
    ax = axes[0, 0]
    blink_left = df[df['is_blink_left']]
    blink_right = df[df['is_blink_right']]
    
    ax.scatter(blink_left['time_seconds'], np.ones(len(blink_left)), 
               color='blue', s=20, alpha=0.6, label='Left eye blinks')
    ax.scatter(blink_right['time_seconds'], np.ones(len(blink_right))*1.1, 
               color='red', s=20, alpha=0.6, label='Right eye blinks')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_title('Blink Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Blink frequency over time
    ax = axes[0, 1]
    window_size = 5000  # 10-second windows
    blink_freq_left = []
    blink_freq_right = []
    time_windows = []
    
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        freq_left = window['is_blink_left'].sum() / (window_size / SAMPLING_RATE)
        freq_right = window['is_blink_right'].sum() / (window_size / SAMPLING_RATE)
        blink_freq_left.append(freq_left)
        blink_freq_right.append(freq_right)
        time_windows.append(window['time_seconds'].mean())
    
    ax.plot(time_windows, blink_freq_left, 'b-', label='Left eye', linewidth=2)
    ax.plot(time_windows, blink_freq_right, 'r-', label='Right eye', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Blinks per second')
    ax.set_title('Blink Frequency Over Time (10s windows)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Missing data periods
    ax = axes[1, 0]
    missing_left = df['x_left'].isna() | df['y_left'].isna()
    missing_right = df['x_right'].isna() | df['y_right'].isna()
    
    # Plot missing data as horizontal bars
    y_pos = 0
    for eye, missing in [('Left', missing_left), ('Right', missing_right)]:
        # Find continuous missing periods
        diff = missing.astype(int).diff()
        starts = df.loc[diff == 1, 'time_seconds'].values
        ends = df.loc[diff == -1, 'time_seconds'].values
        
        # Handle edge cases
        if missing.iloc[0]:
            starts = np.concatenate([[df['time_seconds'].iloc[0]], starts])
        if missing.iloc[-1]:
            ends = np.concatenate([ends, [df['time_seconds'].iloc[-1]]])
        
        # Plot bars
        for start, end in zip(starts, ends):
            ax.barh(y_pos, end - start, left=start, height=0.8, 
                    color='blue' if eye == 'Left' else 'red', alpha=0.5)
        y_pos += 1
    
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Missing Data Periods')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Binocular coordination
    ax = axes[1, 1]
    both_blinks = df['is_blink_left'] & df['is_blink_right']
    left_only = df['is_blink_left'] & ~df['is_blink_right']
    right_only = ~df['is_blink_left'] & df['is_blink_right']
    
    categories = ['Both eyes', 'Left only', 'Right only']
    counts = [both_blinks.sum(), left_only.sum(), right_only.sum()]
    
    bars = ax.bar(categories, counts, color=['purple', 'blue', 'red'])
    ax.set_ylabel('Number of samples')
    ax.set_title('Blink Coordination')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    plt.suptitle('Blink Analysis', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_saccade_analysis(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Extract and analyze saccade patterns.
    
    Args:
        df: Dataframe with saccade data
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract saccade events
    saccade_events = []
    in_saccade = False
    
    for idx, row in df.iterrows():
        both_saccade = row['is_saccade_left'] and row['is_saccade_right']
        
        if both_saccade and not in_saccade:
            in_saccade = True
            start_idx = idx
            start_x_left = row['x_left']
            start_y_left = row['y_left']
            start_time = row['time_seconds']
            
        elif not both_saccade and in_saccade:
            if pd.notna(start_x_left) and pd.notna(row['x_left']):
                distance = np.sqrt((row['x_left'] - start_x_left)**2 + 
                                 (row['y_left'] - start_y_left)**2)
                duration = row['time_seconds'] - start_time
                
                saccade_events.append({
                    'start_time': start_time,
                    'duration_ms': duration * 1000,
                    'amplitude': distance,
                    'start_x': start_x_left,
                    'start_y': start_y_left,
                    'end_x': row['x_left'],
                    'end_y': row['y_left']
                })
            in_saccade = False
    
    saccades_df = pd.DataFrame(saccade_events)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Saccade amplitudes over time
    ax = axes[0, 0]
    if len(saccades_df) > 0:
        ax.scatter(saccades_df['start_time'], saccades_df['amplitude'],
                   alpha=0.6, s=20, c='darkblue')
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Small')
        ax.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Large')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Saccade Amplitude (pixels)')
    ax.set_title('Saccade Amplitudes Throughout Recording')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Saccade duration vs amplitude
    ax = axes[0, 1]
    if len(saccades_df) > 0:
        ax.scatter(saccades_df['duration_ms'], saccades_df['amplitude'],
                   alpha=0.6, s=20)
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Amplitude (pixels)')
        ax.set_title('Saccade Main Sequence')
    ax.grid(True, alpha=0.3)
    
    # 3. Saccade directions (vector plot)
    ax = axes[1, 0]
    if len(saccades_df) > 0:
        # Sample saccades to avoid overcrowding
        sample_idx = np.random.choice(len(saccades_df), 
                                    min(100, len(saccades_df)), 
                                    replace=False)
        for idx in sample_idx:
            sac = saccades_df.iloc[idx]
            dx = sac['end_x'] - sac['start_x']
            dy = sac['end_y'] - sac['start_y']
            ax.arrow(sac['start_x'], sac['start_y'], dx, dy,
                    head_width=20, head_length=30, fc='blue', 
                    ec='blue', alpha=0.5, length_includes_head=True)
    
    screen_rect = Rectangle((0, 0), SCREEN_WIDTH, SCREEN_HEIGHT,
                           fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(screen_rect)
    ax.set_xlim(-100, SCREEN_WIDTH + 100)
    ax.set_ylim(-100, SCREEN_HEIGHT + 100)
    ax.invert_yaxis()
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('Y coordinate (pixels)')
    ax.set_title(f'Saccade Vectors (n={min(100, len(saccades_df))} sampled)')
    ax.set_aspect('equal')
    
    # 4. Saccade statistics
    ax = axes[1, 1]
    if len(saccades_df) > 0:
        stats_text = f"Total saccades: {len(saccades_df)}\n\n"
        stats_text += f"Amplitude (pixels):\n"
        stats_text += f"  Mean: {saccades_df['amplitude'].mean():.1f}\n"
        stats_text += f"  Median: {saccades_df['amplitude'].median():.1f}\n"
        stats_text += f"  Std: {saccades_df['amplitude'].std():.1f}\n\n"
        stats_text += f"Duration (ms):\n"
        stats_text += f"  Mean: {saccades_df['duration_ms'].mean():.1f}\n"
        stats_text += f"  Median: {saccades_df['duration_ms'].median():.1f}\n"
        stats_text += f"  Std: {saccades_df['duration_ms'].std():.1f}\n\n"
        stats_text += f"Rate: {len(saccades_df) / df['time_seconds'].max():.2f} saccades/s"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')
    ax.axis('off')
    ax.set_title('Saccade Statistics')
    
    plt.suptitle(f'Binocular Saccade Analysis (n={len(saccades_df)} saccades)', 
                 fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_eye_stabilization_effect(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    time_window: Tuple[float, float] = (30, 35),
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize the effect of eye stabilization on bouncy data.

    Args:
        df_before: DataFrame before stabilization
        df_after: DataFrame after stabilization
        time_window: Time window to zoom in on (seconds)
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Time mask for zoomed view
    mask = (df_before['time_seconds'] >= time_window[0]) & \
           (df_before['time_seconds'] <= time_window[1])

    # Define clear color scheme for before/after comparison
    colors_before = {'left': '#ADD8E6', 'right': '#FFB6C1'}  # Light blue/pink for before
    colors_after = {'left': '#0000FF', 'right': '#FF0000'}   # Dark blue/red for after

    # 1. X coordinates before/after
    ax = axes[0, 0]
    ax.plot(df_before.loc[mask, 'time_seconds'], df_before.loc[mask, 'x_left'],
            color=colors_before['left'], alpha=0.8, linewidth=2, label='Left before')
    ax.plot(df_before.loc[mask, 'time_seconds'], df_before.loc[mask, 'x_right'],
            color=colors_before['right'], alpha=0.8, linewidth=2, label='Right before')
    ax.plot(df_after.loc[mask, 'time_seconds'], df_after.loc[mask, 'x_left'],
            color=colors_after['left'], alpha=1, linewidth=3, label='Left after')
    ax.plot(df_after.loc[mask, 'time_seconds'], df_after.loc[mask, 'x_right'],
            color=colors_after['right'], alpha=1, linewidth=3, label='Right after')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('X coordinate (pixels)')
    ax.set_title(f'Horizontal Eye Position ({time_window[0]}-{time_window[1]}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Y coordinates before/after
    ax = axes[0, 1]
    ax.plot(df_before.loc[mask, 'time_seconds'], df_before.loc[mask, 'y_left'],
            color=colors_before['left'], alpha=0.8, linewidth=2, label='Left before')
    ax.plot(df_before.loc[mask, 'time_seconds'], df_before.loc[mask, 'y_right'],
            color=colors_before['right'], alpha=0.8, linewidth=2, label='Right before')
    ax.plot(df_after.loc[mask, 'time_seconds'], df_after.loc[mask, 'y_left'],
            color=colors_after['left'], alpha=1, linewidth=3, label='Left after')
    ax.plot(df_after.loc[mask, 'time_seconds'], df_after.loc[mask, 'y_right'],
            color=colors_after['right'], alpha=1, linewidth=3, label='Right after')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Y coordinate (pixels)')
    ax.set_title(f'Vertical Eye Position ({time_window[0]}-{time_window[1]}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Noise analysis
    ax = axes[1, 0]
    # Calculate noise (second derivative) in sliding windows
    window_size = 50
    noise_left_before = df_before['x_left'].diff().diff().rolling(window_size).std()
    noise_right_before = df_before['x_right'].diff().diff().rolling(window_size).std()
    noise_left_after = df_after['x_left'].diff().diff().rolling(window_size).std()
    noise_right_after = df_after['x_right'].diff().diff().rolling(window_size).std()

    ax.plot(df_before['time_seconds'], noise_left_before, 
            color=colors_before['left'], alpha=0.8, linewidth=2, label='Left before')
    ax.plot(df_before['time_seconds'], noise_right_before, 
            color=colors_before['right'], alpha=0.8, linewidth=2, label='Right before')
    ax.plot(df_after['time_seconds'], noise_left_after, 
            color=colors_after['left'], alpha=1, linewidth=3, label='Left after')
    ax.plot(df_after['time_seconds'], noise_right_after, 
            color=colors_after['right'], alpha=1, linewidth=3, label='Right after')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Movement Noise (pixels)')
    ax.set_title('Eye Movement Noise Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1])

    # 4. Eye position relationship
    ax = axes[1, 1]
    valid_before = df_before[df_before['both_eyes_valid']]
    valid_after = df_after[df_after['both_eyes_valid']]

    # Plot eye relationship with clear color distinction
    ax.scatter(valid_before['x_left'], valid_before['x_right'],
               alpha=0.4, s=3, c='red', label='Before', edgecolors='none')
    ax.scatter(valid_after['x_left'], valid_after['x_right'],
               alpha=0.7, s=3, c='blue', label='After', edgecolors='none')

    # Add diagonal line (perfect alignment)
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, 'k--', alpha=0.7, linewidth=2, label='Perfect alignment')

    # Add anatomical constraint line
    ax.plot(xlim, [x + 40 for x in xlim], 'orange', linestyle='--', alpha=0.7, linewidth=2,
            label='Min eye distance (40px)')

    ax.set_xlabel('Left Eye X (pixels)')
    ax.set_ylabel('Right Eye X (pixels)')
    ax.set_title('Eye Position Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle('Eye Stabilization and Anatomical Constraint Effects', fontsize=16)
    plt.tight_layout()

    return fig


def plot_individual_preprocessing_steps(
    pipeline_stages: List[pd.DataFrame],
    pipeline_titles: List[str],
    figsize: Tuple[int, int] = (14, 10),
    save_dir: str = None
) -> List[plt.Figure]:
    """
    Create individual plots for each preprocessing step with gaze trajectory
    on top and comprehensive quality metrics on bottom.

    Args:
        pipeline_stages: List of dataframes at each pipeline stage
        pipeline_titles: List of titles for each stage
        figsize: Figure size for each plot
        save_dir: Directory to save individual plots

    Returns:
        figs: List of matplotlib figures
    """
    from src.quality import calculate_comprehensive_quality_metrics
    from src.quality import ANALYSIS_PARAMS
    
    figures = []

    for i, (df, title) in enumerate(zip(pipeline_stages, pipeline_titles)):
        # Create figure with 2 subplots
        fig, (ax_traj, ax_quality) = plt.subplots(2, 1, figsize=figsize,
                                                  gridspec_kw={'height_ratios': [2, 1]})

        # Top plot: Gaze trajectory
        sample_rate = PREPROCESSING_PARAMS['plot_sample_rate']
        sampled_df = df[::sample_rate]

        # Plot valid eye data
        valid_left = sampled_df.dropna(subset=['x_left', 'y_left'])
        valid_right = sampled_df.dropna(subset=['x_right', 'y_right'])

        if len(valid_left) > 0:
            ax_traj.scatter(valid_left['x_left'], valid_left['y_left'],
                           c='blue', alpha=0.6, s=3, label='Left eye', edgecolors='none')

        if len(valid_right) > 0:
            ax_traj.scatter(valid_right['x_right'], valid_right['y_right'],
                           c='red', alpha=0.6, s=3, label='Right eye', edgecolors='none')

        # Add screen boundaries
        screen_rect = Rectangle((0, 0), SCREEN_WIDTH, SCREEN_HEIGHT,
                               fill=False, edgecolor='black', linewidth=3, label='Screen')
        ax_traj.add_patch(screen_rect)

        # Set limits and formatting
        margin = 200
        ax_traj.set_xlim(-margin, SCREEN_WIDTH + margin)
        ax_traj.set_ylim(-margin, SCREEN_HEIGHT + margin)
        ax_traj.invert_yaxis()
        ax_traj.set_xlabel('X coordinate (pixels)')
        ax_traj.set_ylabel('Y coordinate (pixels)')
        ax_traj.set_title(f'{title} - Gaze Trajectory')
        ax_traj.legend()
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect('equal', adjustable='box')

        # Add data quality stats
        total_samples = len(df)
        valid_samples = df['both_eyes_valid'].sum() if 'both_eyes_valid' in df.columns else 0
        validity_pct = (valid_samples / total_samples * 100) if total_samples > 0 else 0

        ax_traj.text(0.02, 0.98, f'Valid samples: {valid_samples:,}/{total_samples:,} ({validity_pct:.1f}%)',
                    transform=ax_traj.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Bottom plot: Comprehensive Quality Metrics
        if 'both_eyes_valid' in df.columns and 'time_seconds' in df.columns:
            try:
                # Calculate quality metrics for this stage
                print(f"🔍 Calculating quality metrics for {title}...")
                quality_metrics = calculate_comprehensive_quality_metrics(df)

                # Create sliding window analysis
                window_size = ANALYSIS_PARAMS['window_size']
                step_size = window_size // 4
                time_points = []
                basic_validity_vals = []
                alignment_quality_vals = []
                plausibility_vals = []
                snr_vals = []
                composite_vals = []

                for j in range(0, len(quality_metrics) - window_size, step_size):
                    window = quality_metrics.iloc[j:j+window_size]

                    # Time point (middle of window)
                    time_points.append(window['time_seconds'].iloc[window_size//2])

                    # Calculate averages for each metric
                    basic_validity_vals.append(window['basic_validity'].mean() * 100)
                    alignment_quality_vals.append(window['alignment_quality'].mean() * 100)
                    plausibility_vals.append(window['plausibility_score'].mean() * 100)
                    snr_vals.append(window['snr_score'].mean() * 100)
                    composite_vals.append(window['composite_quality'].mean() * 100)

                if time_points:
                    # Plot all quality metrics with distinct styling
                    ax_quality.plot(time_points, basic_validity_vals,
                                   linewidth=2, label='Basic Validity', color='#2E86AB', alpha=0.9)
                    ax_quality.plot(time_points, alignment_quality_vals,
                                   linewidth=1.5, label='Binocular Alignment', color='#A23B72', alpha=0.8)
                    ax_quality.plot(time_points, plausibility_vals,
                                   linewidth=1.5, label='Physiological Plausibility', color='#F18F01', alpha=0.8)
                    ax_quality.plot(time_points, snr_vals,
                                   linewidth=1, label='Signal-to-Noise Ratio', color='#C73E1D', alpha=0.7)

                    # Composite score with special styling
                    ax_quality.plot(time_points, composite_vals,
                                   linewidth=2.5, label='Composite Quality', color='#1B5E20', alpha=0.9, linestyle='--')

                    # Add quality thresholds
                    ax_quality.axhline(y=95, color='green', linestyle=':', alpha=0.6)
                    ax_quality.axhline(y=85, color='orange', linestyle=':', alpha=0.6)
                    ax_quality.axhline(y=70, color='red', linestyle=':', alpha=0.6)

            except Exception as e:
                print(f"⚠️ Could not calculate quality metrics for {title}: {str(e)}")
                # Fallback to simple validity plot
                window_size = ANALYSIS_PARAMS['window_size']
                validity_over_time = []
                time_points = []

                for j in range(0, len(df) - window_size, window_size // 4):
                    window = df.iloc[j:j+window_size]
                    if len(window) > 0:
                        validity_percent = window['both_eyes_valid'].sum() / len(window) * 100
                        validity_over_time.append(validity_percent)
                        time_points.append(window['time_seconds'].mean())

                if time_points:
                    ax_quality.plot(time_points, validity_over_time, 'g-', linewidth=2, alpha=0.8)
                    ax_quality.fill_between(time_points, validity_over_time, alpha=0.3, color='green')

        ax_quality.set_xlabel('Time (seconds)')
        ax_quality.set_ylabel('Quality Score (%)')
        ax_quality.set_title('Comprehensive Data Quality Assessment Over Time')
        ax_quality.set_ylim(0, 105)
        ax_quality.legend(loc='lower left', fontsize=8, ncol=2)
        ax_quality.grid(True, alpha=0.3)

        # Overall figure title
        fig.suptitle(f'Step {i+1}: {title}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save individual figure if directory provided
        if save_dir:
            filename = f'step_{i+1:02d}_{title.lower().replace(" ", "_").replace("-", "_")}.png'
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=PREPROCESSING_PARAMS['figure_dpi'], bbox_inches='tight')
            print(f"✓ Saved: {filename}")

        figures.append(fig)

    return figures


def plot_comprehensive_quality_trends(
    quality_metrics: pd.DataFrame, 
    window_size: int = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Plot multiple quality trends in sliding windows on one comprehensive graph.
    
    Args:
        quality_metrics: DataFrame with quality metrics
        window_size: Sliding window size (default from ANALYSIS_PARAMS)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    from src.quality import ANALYSIS_PARAMS
    if window_size is None:
        window_size = ANALYSIS_PARAMS['window_size']
    
    print(f"📊 Creating comprehensive quality visualization (window size: {window_size/SAMPLING_RATE:.1f}s)...")
    
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
    
    # Create the comprehensive plot
    fig = plt.figure(figsize=figsize)
    
    # Plot all quality metrics with distinct styling
    plt.plot(time_points, basic_validity_vals, 
             linewidth=2.5, label='Basic Validity', color='#2E86AB', alpha=0.9)
    plt.plot(time_points, alignment_quality_vals, 
             linewidth=2, label='Binocular Alignment', color='#A23B72', alpha=0.8)
    plt.plot(time_points, plausibility_vals, 
             linewidth=2, label='Physiological Plausibility', color='#F18F01', alpha=0.8)
    plt.plot(time_points, snr_vals, 
             linewidth=1.5, label='Signal-to-Noise Ratio', color='#C73E1D', alpha=0.7)
    
    # Composite score with special styling
    plt.plot(time_points, composite_vals, 
             linewidth=3, label='Composite Quality Score', color='#1B5E20', alpha=0.9, linestyle='--')
    
    # Add quality thresholds
    plt.axhline(y=95, color='green', linestyle=':', alpha=0.6, label='Excellent (95%+)')
    plt.axhline(y=85, color='orange', linestyle=':', alpha=0.6, label='Good (85%+)')
    plt.axhline(y=70, color='red', linestyle=':', alpha=0.6, label='Acceptable (70%+)')
    
    # Formatting
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Quality Score (%)', fontsize=12)
    plt.title('Comprehensive Data Quality Assessment\n(Multiple Quality Dimensions Over Time)', 
              fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.xlim(0, 90)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    # Add informative annotation
    plt.text(0.02, 0.98, 
             f'Window size: {window_size/SAMPLING_RATE:.1f}s | Overlap: 75%', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig


def plot_social_attention_summary(metrics: dict, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a summary plot for social attention metrics.
    
    Args:
        metrics: Social attention metrics dictionary
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Social vs. Non-social pie chart
    ax = axes[0, 0]
    social_pct = metrics.get('social_attention_percent', 0)
    nonsocial_pct = 100 - social_pct
    
    ax.pie([social_pct, nonsocial_pct], 
           labels=['Social', 'Non-Social'],
           autopct='%1.1f%%',
           colors=['#5DA5DA', '#FAA43A'],
           startangle=90)
    ax.set_title('Social vs. Non-Social Attention')
    
    # 2. ROI distribution bar chart
    ax = axes[0, 1]
    roi_dist = metrics.get('roi_distribution', {})
    
    if roi_dist:
        # Sort by social (color grouping) then by percentage
        sorted_rois = sorted(roi_dist.items(), 
                            key=lambda x: (x[0] not in ['Head', 'Hand', 'Torso'], -x[1]))
        
        # Filter out any None values that might cause errors
        sorted_rois = [(label, value) for label, value in sorted_rois if label is not None and value is not None]
        
        if sorted_rois:  # Check if we still have valid data after filtering
            labels = [str(roi[0]) for roi in sorted_rois]  # Convert to strings to avoid type issues
            values = [roi[1] for roi in sorted_rois]
            
            colors = []
            for label in labels:
                if label in ['Head', 'Hand', 'Torso']:
                    colors.append('#5DA5DA')  # Blue for social
                else:
                    colors.append('#FAA43A')  # Orange for non-social
            
            bars = ax.bar(labels, values, color=colors)
            ax.set_title('ROI Distribution')
            ax.set_ylabel('Percentage (%)')
            if values:  # Check if values list is not empty
                ax.set_ylim(0, max(values) * 1.2)
            
            # Rotate labels if there are many
            if len(labels) > 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No valid ROI distribution data', 
                    ha='center', va='center', 
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No ROI distribution data', 
                ha='center', va='center', 
                transform=ax.transAxes)
    
    # 3. Fixation durations by ROI
    ax = axes[1, 0]
    fixation_durations = metrics.get('fixation_patterns', {}).get('fixation_duration_by_roi', {})
    
    if fixation_durations:
        # Sort by social (color grouping) then by duration
        sorted_durations = sorted(fixation_durations.items(), 
                                key=lambda x: (x[0] not in ['Head', 'Hand', 'Torso'], -x[1]))
        
        # Filter out any None values that might cause errors
        sorted_durations = [(label, value) for label, value in sorted_durations if label is not None and value is not None]
        
        if sorted_durations:  # Check if we still have valid data after filtering
            labels = [str(d[0]) for d in sorted_durations]  # Convert to strings to avoid type issues
            values = [d[1] for d in sorted_durations]
            
            colors = []
            for label in labels:
                if label in ['Head', 'Hand', 'Torso']:
                    colors.append('#5DA5DA')  # Blue for social
                else:
                    colors.append('#FAA43A')  # Orange for non-social
            
            bars = ax.bar(labels, values, color=colors)
            ax.set_title('Average Fixation Duration by ROI')
            ax.set_ylabel('Duration (seconds)')
            if values:  # Check if values list is not empty
                ax.set_ylim(0, max(values) * 1.2)
            
            # Rotate labels if there are many
            if len(labels) > 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No valid fixation duration data', 
                    ha='center', va='center', 
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No fixation duration data', 
                ha='center', va='center', 
                transform=ax.transAxes)
    
    # 4. Transition matrix/metrics
    ax = axes[1, 1]
    transitions = metrics.get('transitions', {})
    
    # Check for minimum required transition counts to make a meaningful matrix
    social_to_social = transitions.get('social_to_social', 0)
    social_to_nonsocial = transitions.get('social_to_nonsocial', 0)
    nonsocial_to_social = transitions.get('nonsocial_to_social', 0)
    nonsocial_to_nonsocial = transitions.get('nonsocial_to_nonsocial', 0)
    
    total_transitions = social_to_social + social_to_nonsocial + nonsocial_to_social + nonsocial_to_nonsocial
    
    if total_transitions >= 10:  # Only show if we have enough transitions for meaningful data
        # Create a 2x2 transition matrix
        transition_matrix = np.zeros((2, 2))
        
        # Fill the matrix
        # [social->social, social->nonsocial]
        # [nonsocial->social, nonsocial->nonsocial]
        transition_matrix[0, 0] = social_to_social
        transition_matrix[0, 1] = social_to_nonsocial
        transition_matrix[1, 0] = nonsocial_to_social
        transition_matrix[1, 1] = nonsocial_to_nonsocial
        
        # Normalize by row to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.zeros_like(transition_matrix)
        
        # Only normalize rows with non-zero sums
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                normalized_matrix[i] = transition_matrix[i] / row_sums[i]
            else:
                # For rows with no transitions, use NaN to indicate no data
                normalized_matrix[i] = np.nan
        
        # Plot heatmap
        im = ax.imshow(normalized_matrix, cmap='Blues', vmin=0, vmax=1)
        
        # Add labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Social', 'Non-Social'])
        ax.set_yticklabels(['Social', 'Non-Social'])
        ax.set_xlabel('To')
        ax.set_ylabel('From')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                if np.isnan(normalized_matrix[i, j]):
                    text = ax.text(j, i, "N/A", ha="center", va="center", color="black")
                else:
                    text = ax.text(j, i, f'{normalized_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black")
                    
                    # Add raw counts in parentheses
                    raw_count = transition_matrix[i, j]
                    if raw_count > 0:
                        ax.text(j, i+0.3, f'({int(raw_count)})', 
                               ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Transition Probability')
        
        ax.set_title(f'Gaze Transition Probabilities\n(Total: {int(total_transitions)})')
    else:
        ax.text(0.5, 0.5, 'Insufficient transition data\n(Need at least 10 transitions)', 
                ha="center", va="center", 
                transform=ax.transAxes)
    
    # Add overall title
    fig.suptitle('Social Attention Analysis Summary', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_temporal_social_attention(metrics: dict, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot social attention over time.
    
    Args:
        metrics: Social attention metrics dictionary
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get temporal data
    temporal = metrics.get('temporal_patterns', {})
    rolling_data = temporal.get('rolling_social_attention', [])
    
    if rolling_data:
        # Extract data points
        times = [d['time_seconds'] for d in rolling_data]
        social_pcts = [d['social_attention_percent'] for d in rolling_data]
        
        # Plot rolling social attention
        ax.plot(times, social_pcts, 'b-', linewidth=2, marker='o', markersize=4)
        
        # Add reference lines
        ax.axhline(y=metrics.get('social_attention_percent', 0), 
                  color='r', linestyle='--', 
                  label=f'Overall: {metrics.get("social_attention_percent", 0):.1f}%')
        
        # Highlight social engagement sequences if available
        sequences = temporal.get('social_engagement_sequences', [])
        if sequences and 'df_with_roi' in metrics:
            df = metrics['df_with_roi']
            if 'time_seconds' in df.columns:
                for seq in sequences:
                    if len(seq) > 0:
                        if seq[0] in df.index and seq[-1] in df.index:
                            start_time = df.loc[seq[0], 'time_seconds']
                            end_time = df.loc[seq[-1], 'time_seconds']
                            ax.axvspan(start_time, end_time, alpha=0.2, color='green')
        
        # Add annotations
        if 'time_to_first_social_look' in metrics and metrics['time_to_first_social_look'] is not None:
            first_social = metrics['time_to_first_social_look']
            ax.axvline(x=first_social, color='green', linestyle='-.',
                      label=f'First social look: {first_social:.1f}s')
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Social Attention (%)')
        ax.set_title('Social Attention Development Over Time')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No temporal data available', 
                ha='center', va='center', 
                transform=ax.transAxes)
    
    plt.tight_layout()
    
    return fig


def visualize_gaze_on_roi(df: pd.DataFrame, frame_number: int, roi_data: dict, 
                         screen_width: int = SCREEN_WIDTH, screen_height: int = SCREEN_HEIGHT,
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize gaze points on ROI for a specific frame.
    
    Args:
        df: Dataframe with gaze and ROI data
        frame_number: Video frame number to visualize
        roi_data: ROI data dictionary
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a frame showing the screen
    rect = Rectangle((0, 0), screen_width, screen_height, 
                    fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Get ROIs for this frame
    frame_key = str(int(frame_number))
    if frame_key in roi_data:
        # Draw ROIs
        for roi in roi_data[frame_key]:
            # Get polygon coordinates
            if 'coordinates' in roi:
                coords = roi['coordinates']
                x_coords = [c['x'] * screen_width for c in coords]
                y_coords = [c['y'] * screen_height for c in coords]
                
                # Determine color based on label
                if roi['label'] in ['Head', 'Hand', 'Torso']:
                    color = 'blue'
                    alpha = 0.3
                else:
                    color = 'orange'
                    alpha = 0.2
                
                # Draw polygon
                polygon = plt.Polygon(list(zip(x_coords, y_coords)), 
                                     closed=True, fill=True, 
                                     alpha=alpha, color=color)
                ax.add_patch(polygon)
                
                # Add label at centroid
                centroid_x = sum(x_coords) / len(x_coords)
                centroid_y = sum(y_coords) / len(y_coords)
                ax.text(centroid_x, centroid_y, roi['label'], 
                       ha='center', va='center', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Filter data for this frame
    frame_data = df[df['frame_number'] == frame_number]
    
    # Plot gaze points
    for eye in ['left', 'right']:
        x_col = f'x_{eye}'
        y_col = f'y_{eye}'
        
        valid_gaze = frame_data.dropna(subset=[x_col, y_col])
        
        if len(valid_gaze) > 0:
            ax.scatter(valid_gaze[x_col], valid_gaze[y_col], 
                      label=f'{eye.capitalize()} eye',
                      s=100, alpha=0.7,
                      color='red' if eye == 'right' else 'green')
    
    # Set labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Gaze and ROIs - Frame {frame_number}')
    
    # Set limits with margin
    margin = 50
    ax.set_xlim(-margin, screen_width + margin)
    ax.set_ylim(-margin, screen_height + margin)
    
    # Invert y-axis (0 at top for screen coordinates)
    ax.invert_yaxis()
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    return fig