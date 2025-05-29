"""
Feature extraction functions for eye-tracking data analysis.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
import json

# Configuration
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1024
SAMPLING_RATE = 500  # Hz

# Analysis parameters
ANALYSIS_PARAMS = {
    'min_valid_data': 0.60,        # minimum 60% valid data required
    'window_size': 500,            # 1-second windows for temporal analysis
}


def extract_basic_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract basic features from preprocessed eye-tracking data.
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        features: Dictionary of extracted features
    """
    features = {}
    
    # Data quality features
    features['valid_samples_ratio'] = df['both_eyes_valid'].sum() / len(df)
    features['total_duration_seconds'] = df['time_seconds'].max()
    
    # Fixation features
    features['fixation_ratio_left'] = df['is_fixation_left'].sum() / len(df)
    features['fixation_ratio_right'] = df['is_fixation_right'].sum() / len(df)
    features['fixation_ratio_binocular'] = (
        (df['is_fixation_left'] & df['is_fixation_right']).sum() / len(df)
    )
    
    # Saccade features
    features['saccade_ratio_left'] = df['is_saccade_left'].sum() / len(df)
    features['saccade_ratio_right'] = df['is_saccade_right'].sum() / len(df)
    features['saccade_ratio_binocular'] = (
        (df['is_saccade_left'] & df['is_saccade_right']).sum() / len(df)
    )
    
    # Blink features
    features['blink_count_left'] = df['is_blink_left'].sum() / SAMPLING_RATE  # per second
    features['blink_count_right'] = df['is_blink_right'].sum() / SAMPLING_RATE
    
    # Disparity features (only for valid samples)
    valid_mask = df['both_eyes_valid']
    if valid_mask.sum() > 0:
        features['disparity_mean'] = df.loc[valid_mask, 'disparity_total'].mean()
        features['disparity_std'] = df.loc[valid_mask, 'disparity_total'].std()
        features['disparity_median'] = df.loc[valid_mask, 'disparity_total'].median()
        features['disparity_95percentile'] = df.loc[valid_mask, 'disparity_total'].quantile(0.95)
    
    # Velocity features
    for eye in ['left', 'right']:
        vel_col = f'velocity_{eye}'
        if vel_col in df.columns:
            valid_vel = df[vel_col].dropna()
            if len(valid_vel) > 0:
                features[f'velocity_mean_{eye}'] = valid_vel.mean()
                features[f'velocity_std_{eye}'] = valid_vel.std()
                features[f'velocity_median_{eye}'] = valid_vel.median()
    
    # Pupil features
    for eye in ['left', 'right']:
        pupil_col = f'pupil_{eye}'
        if pupil_col in df.columns:
            valid_pupil = df[pupil_col].dropna()
            if len(valid_pupil) > 0:
                features[f'pupil_mean_{eye}'] = valid_pupil.mean()
                features[f'pupil_std_{eye}'] = valid_pupil.std()
                features[f'pupil_cv_{eye}'] = valid_pupil.std() / valid_pupil.mean()
    
    # Spatial distribution features
    for eye in ['left', 'right']:
        x_col, y_col = f'x_{eye}', f'y_{eye}'
        valid_data = df[[x_col, y_col]].dropna()
        
        if len(valid_data) > 0:
            # Coverage of screen
            x_range = valid_data[x_col].max() - valid_data[x_col].min()
            y_range = valid_data[y_col].max() - valid_data[y_col].min()
            features[f'screen_coverage_{eye}'] = (x_range * y_range) / (SCREEN_WIDTH * SCREEN_HEIGHT)
            
            # Center of mass
            features[f'center_x_{eye}'] = valid_data[x_col].mean()
            features[f'center_y_{eye}'] = valid_data[y_col].mean()
            
            # Spread (standard deviation)
            features[f'spread_x_{eye}'] = valid_data[x_col].std()
            features[f'spread_y_{eye}'] = valid_data[y_col].std()
    
    return features


def extract_temporal_features(
    df: pd.DataFrame,
    window_size: int = None
) -> pd.DataFrame:
    """
    Extract temporal features using sliding windows.
    
    Args:
        df: Preprocessed dataframe
        window_size: Window size in samples (default from config)
        
    Returns:
        temporal_features: DataFrame with temporal features
    """
    if window_size is None:
        window_size = ANALYSIS_PARAMS['window_size']
    
    temporal_features = []
    
    # Slide window through data
    for i in range(0, len(df) - window_size, window_size // 2):
        window = df.iloc[i:i+window_size]
        
        # Extract features for this window
        window_features = {
            'window_start': i,
            'window_end': i + window_size,
            'time_start': window['time_seconds'].iloc[0],
            'time_end': window['time_seconds'].iloc[-1],
        }
        
        # Add basic features for this window
        basic_features = extract_basic_features(window)
        window_features.update(basic_features)
        
        temporal_features.append(window_features)
    
    return pd.DataFrame(temporal_features)


def create_feature_matrix(
    preprocessed_files: List[str],
    labels: Optional[Dict[str, int]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix from multiple preprocessed files.
    
    Args:
        preprocessed_files: List of preprocessed CSV files
        labels: Optional dictionary mapping subject_id to label
        
    Returns:
        X: Feature matrix
        y: Labels (if provided)
    """
    all_features = []
    all_labels = []
    
    for file_path in preprocessed_files:
        # Load preprocessed data
        df = pd.read_csv(file_path)
        
        # Extract subject ID
        subject_id = os.path.basename(file_path).split('_')[0]
        
        # Extract features
        features = extract_basic_features(df)
        features['subject_id'] = subject_id
        
        all_features.append(features)
        
        # Add label if available
        if labels and subject_id in labels:
            all_labels.append(labels[subject_id])
    
    # Create feature matrix
    X = pd.DataFrame(all_features)
    X = X.set_index('subject_id')
    
    # Create label series
    y = pd.Series(all_labels, index=X.index) if all_labels else None
    
    return X, y


class ROIAnalyzer:
    """
    Region of Interest (ROI) Analyzer for social attention metrics.
    Maps eye gaze coordinates to semantically meaningful regions and calculates 
    social attention metrics.
    """
    
    def __init__(self, roi_file_path: str, screen_width: int = SCREEN_WIDTH, screen_height: int = SCREEN_HEIGHT):
        """
        Initialize ROI Analyzer with ROI file.
        
        Args:
            roi_file_path: Path to ROI JSON file
            screen_width: Screen width in pixels (default from config)
            screen_height: Screen height in pixels (default from config)
        """
        self.roi_file_path = roi_file_path
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.roi_data = None
        self.social_categories = ['Face', 'Head', 'Hand', 'Torso']  # Social ROI categories (person-related)
        self.object_categories = ['Couch', 'Bed']  # Non-social ROI categories
        
        # Load ROI data
        self._load_roi_data()
        
    def _load_roi_data(self):
        """Load ROI data from JSON file."""
        try:
            with open(self.roi_file_path, 'r') as f:
                self.roi_data = json.load(f)
            print(f"✅ Loaded ROI data from {self.roi_file_path}")
            print(f"   Number of frames with ROI data: {len(self.roi_data)}")
        except Exception as e:
            print(f"❌ Error loading ROI data: {str(e)}")
            self.roi_data = {}
    
    def _point_in_polygon(self, x: float, y: float, polygon_points: list) -> bool:
        """
        Check if a point is inside a polygon using the ray casting algorithm.
        
        Args:
            x: X coordinate of point
            y: Y coordinate of point
            polygon_points: List of (x,y) coordinates forming the polygon
            
        Returns:
            bool: True if point is inside the polygon, False otherwise
        """
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]['x'], polygon_points[0]['y']
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]['x'], polygon_points[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _point_in_roi(self, x: float, y: float, roi: dict) -> bool:
        """
        Check if a point is inside an ROI polygon.
        
        Args:
            x: X coordinate of point (normalized 0-1)
            y: Y coordinate of point (normalized 0-1)
            roi: ROI dictionary with coordinates
            
        Returns:
            bool: True if point is inside the ROI, False otherwise
        """
        # Convert normalized coordinates to pixels if needed
        if 0 <= x <= 1 and 0 <= y <= 1:
            # Already normalized, use as is
            pass
        else:
            # Convert to normalized
            x = x / self.screen_width
            y = y / self.screen_height
        
        # Check if the point is inside the ROI using ray casting algorithm
        if 'coordinates' in roi:
            return self._point_in_polygon(x, y, roi['coordinates'])
        elif 'bounding_box' in roi:
            # Use bounding box for simpler check
            bb = roi['bounding_box']
            return bb['left'] <= x <= bb['right'] and bb['top'] <= y <= bb['bottom']
        
        return False
    
    def get_roi_for_point(self, x: float, y: float, frame_number: int) -> dict:
        """
        Find which ROI contains the given point for a specific frame.
        
        Args:
            x: X coordinate
            y: Y coordinate
            frame_number: Video frame number
            
        Returns:
            dict: ROI information or None if no ROI contains the point
        """
        # If frame_number is float, convert to int
        if isinstance(frame_number, float):
            frame_number = int(frame_number)
            
        frame_key = str(frame_number)
        
        # If the exact frame doesn't exist, try to find the closest frame
        if frame_key not in self.roi_data:
            # Convert all keys to integers for comparison
            numeric_keys = [int(k) for k in self.roi_data.keys() if k.isdigit()]
            if not numeric_keys:
                return None
                
            # Find closest frame
            closest_frame = min(numeric_keys, key=lambda k: abs(k - frame_number))
            frame_key = str(closest_frame)
            
            # If still too far, return None
            if abs(closest_frame - frame_number) > 10:  # Arbitrary threshold
                return None
        
        # Check each ROI in the frame
        for roi in self.roi_data[frame_key]:
            if self._point_in_roi(x, y, roi):
                return {
                    'object_id': roi['object_id'],
                    'label': roi['label'],
                    'is_social': roi['label'] in self.social_categories
                }
        
        # No ROI contains the point
        return None
    
    def analyze_gaze_social_attention(self, df: pd.DataFrame, use_binocular: bool = True) -> dict:
        """
        Analyze social attention in gaze data.
        
        Args:
            df: Preprocessed dataframe with gaze coordinates
            use_binocular: Whether to use both eyes or just one
            
        Returns:
            dict: Dictionary of social attention metrics
        """
        # Ensure we have frame numbers
        if 'frame_number' not in df.columns:
            print("❌ Error: Dataframe must contain 'frame_number' column")
            return {}
        
        # Initialize metrics
        metrics = {
            'total_valid_frames': 0,
            'social_attention_frames': 0,
            'social_attention_percent': 0.0,
            'person_attention_frames': 0,
            'person_attention_percent': 0.0,
            'face_attention_frames': 0,
            'face_attention_percent': 0.0,
            'head_attention_frames': 0,
            'head_attention_percent': 0.0,
            'hand_attention_frames': 0,
            'hand_attention_percent': 0.0,
            'torso_attention_frames': 0,
            'torso_attention_percent': 0.0,
            'object_attention_frames': 0,
            'object_attention_percent': 0.0,
            'roi_distribution': {},
            'transitions': {
                'social_to_social': 0,
                'social_to_nonsocial': 0,
                'nonsocial_to_social': 0,
                'nonsocial_to_nonsocial': 0,
                'total_transitions': 0
            },
            'first_fixation': None,
            'time_to_first_social_look': None
        }
        
        # Track transitions
        previous_roi_type = None  # Can be 'social', 'nonsocial', or None
        previous_roi_label = None  # Keep track of the actual ROI label
        
        # Process each valid frame
        roi_counts = {}
        frames_processed = 0
        
        # For transition analysis
        transitions = []
        
        # For debugging
        roi_matches = 0
        roi_misses = 0
        
        # Make a copy of the dataframe to avoid modifying the original
        df_analysis = df.copy()
        
        # Add columns for ROI information
        df_analysis['roi_id'] = None
        df_analysis['roi_label'] = None
        df_analysis['is_social_roi'] = False
        
        # Loop through each frame
        for idx, row in df_analysis.iterrows():
            # Skip if no frame number
            if pd.isna(row['frame_number']):
                continue
                
            # Determine which eye to use
            if use_binocular and row.get('both_eyes_valid', False):
                # Use average of both eyes
                x = (row['x_left'] + row['x_right']) / 2
                y = (row['y_left'] + row['y_right']) / 2
            elif pd.notna(row.get('x_left')) and pd.notna(row.get('y_left')):
                # Use left eye
                x, y = row['x_left'], row['y_left']
            elif pd.notna(row.get('x_right')) and pd.notna(row.get('y_right')):
                # Use right eye
                x, y = row['x_right'], row['y_right']
            else:
                # No valid gaze data
                continue
            
            # Get ROI for this point
            roi = self.get_roi_for_point(x, y, row['frame_number'])
            
            # Store ROI info in dataframe
            if roi:
                roi_matches += 1
                df_analysis.at[idx, 'roi_id'] = roi['object_id']
                df_analysis.at[idx, 'roi_label'] = roi['label']
                df_analysis.at[idx, 'is_social_roi'] = roi['is_social']
                
                # Count ROI occurrence
                label = roi['label']
                roi_counts[label] = roi_counts.get(label, 0) + 1
                
                # Update metrics
                metrics['total_valid_frames'] += 1
                
                if roi['is_social']:
                    metrics['social_attention_frames'] += 1
                    current_roi_type = 'social'
                    
                    # All social attention in this dataset is person-related
                    metrics['person_attention_frames'] += 1
                    
                    # Track specific body parts
                    if label == 'Face':
                        metrics['face_attention_frames'] = metrics.get('face_attention_frames', 0) + 1
                    elif label == 'Head':
                        metrics['head_attention_frames'] = metrics.get('head_attention_frames', 0) + 1
                    elif label == 'Hand':
                        metrics['hand_attention_frames'] = metrics.get('hand_attention_frames', 0) + 1
                    elif label == 'Torso':
                        metrics['torso_attention_frames'] = metrics.get('torso_attention_frames', 0) + 1
                else:
                    current_roi_type = 'nonsocial'
                    metrics['object_attention_frames'] += 1
                
                # Track transitions between social and non-social content
                # A transition occurs when moving from social to non-social or vice versa
                # We also track when staying within the same category
                if previous_roi_type is not None:
                    # Only count as transition when there's an actual ROI change OR when crossing social boundary
                    if previous_roi_label != label or previous_roi_type != current_roi_type:
                        transition_key = f'{previous_roi_type}_to_{current_roi_type}'
                        metrics['transitions'][transition_key] = metrics['transitions'].get(transition_key, 0) + 1
                        metrics['transitions']['total_transitions'] += 1
                        
                        # Add to transition sequence
                        transitions.append((previous_roi_type, current_roi_type, idx))
                
                previous_roi_type = current_roi_type
                previous_roi_label = label
                
                # First fixation
                if metrics['first_fixation'] is None:
                    metrics['first_fixation'] = label
                    if roi['is_social']:
                        metrics['time_to_first_social_look'] = row.get('time_seconds', idx / SAMPLING_RATE)
            else:
                roi_misses += 1
                # Point is not in any ROI
                df_analysis.at[idx, 'roi_id'] = 'none'
                df_analysis.at[idx, 'roi_label'] = 'none'
                df_analysis.at[idx, 'is_social_roi'] = False
                
                # Don't reset previous_roi_type, as we want to detect transitions 
                # even if there are a few frames without ROI in between
                
                # Only reset if we've had several consecutive frames without ROI
                if roi_misses > 10:  # Arbitrary threshold to avoid false transitions
                    previous_roi_type = None
                    previous_roi_label = None
            
            frames_processed += 1
        
        # Add debug info
        metrics['debug_info'] = {
            'roi_matches': roi_matches,
            'roi_misses': roi_misses,
            'frames_processed': frames_processed
        }
        
        # Calculate percentages
        if metrics['total_valid_frames'] > 0:
            metrics['social_attention_percent'] = metrics['social_attention_frames'] / metrics['total_valid_frames'] * 100
            metrics['person_attention_percent'] = metrics['person_attention_frames'] / metrics['total_valid_frames'] * 100
            metrics['face_attention_percent'] = metrics['face_attention_frames'] / metrics['total_valid_frames'] * 100
            metrics['head_attention_percent'] = metrics['head_attention_frames'] / metrics['total_valid_frames'] * 100
            metrics['hand_attention_percent'] = metrics['hand_attention_frames'] / metrics['total_valid_frames'] * 100
            metrics['torso_attention_percent'] = metrics['torso_attention_frames'] / metrics['total_valid_frames'] * 100
            metrics['object_attention_percent'] = metrics['object_attention_frames'] / metrics['total_valid_frames'] * 100
        
        # ROI distribution
        total_roi_frames = sum(roi_counts.values())
        roi_distribution = {}
        if total_roi_frames > 0:
            for label, count in roi_counts.items():
                roi_distribution[label] = count / total_roi_frames * 100
        metrics['roi_distribution'] = roi_distribution
        
        # Calculate temporal patterns
        metrics['temporal_patterns'] = self._analyze_temporal_patterns(df_analysis)
        
        # Calculate fixation patterns
        metrics['fixation_patterns'] = self._analyze_fixation_patterns(df_analysis)
        
        # Store the modified dataframe with ROI info
        metrics['df_with_roi'] = df_analysis
        
        # Print debugging info
        print(f"📊 ROI Analysis Summary:")
        print(f"   ROI matches: {roi_matches}")
        print(f"   ROI misses: {roi_misses}")
        print(f"   Transitions detected: {metrics['transitions']['total_transitions']}")
        
        return metrics
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, window_size: int = 500) -> dict:
        """
        Analyze temporal patterns of social attention.
        
        Args:
            df: Dataframe with ROI information
            window_size: Window size for rolling metrics
            
        Returns:
            dict: Temporal patterns metrics
        """
        temporal_metrics = {
            'rolling_social_attention': [],
            'social_engagement_persistence': 0,
            'social_engagement_sequences': []
        }
        
        # Rolling social attention
        if 'is_social_roi' in df.columns and 'time_seconds' in df.columns:
            # Calculate rolling metrics
            df_valid = df.dropna(subset=['is_social_roi', 'time_seconds'])
            
            if len(df_valid) > window_size:
                rolling_data = []
                
                for i in range(0, len(df_valid) - window_size, window_size // 2):
                    window = df_valid.iloc[i:i+window_size]
                    social_percent = window['is_social_roi'].sum() / len(window) * 100
                    
                    rolling_data.append({
                        'time_seconds': window['time_seconds'].mean(),
                        'social_attention_percent': social_percent
                    })
                
                temporal_metrics['rolling_social_attention'] = rolling_data
        
        # Find continuous social engagement sequences
        if 'is_social_roi' in df.columns:
            in_social_sequence = False
            current_sequence = []
            sequences = []
            
            for idx, is_social in enumerate(df['is_social_roi']):
                if is_social:
                    if not in_social_sequence:
                        # Start new sequence
                        in_social_sequence = True
                        current_sequence = [idx]
                    else:
                        # Continue sequence
                        current_sequence.append(idx)
                else:
                    if in_social_sequence:
                        # End sequence
                        in_social_sequence = False
                        if len(current_sequence) > 5:  # Minimum sequence length
                            sequences.append(current_sequence)
                        current_sequence = []
            
            # Add last sequence if active
            if in_social_sequence and len(current_sequence) > 5:
                sequences.append(current_sequence)
            
            # Calculate social engagement persistence (average sequence length)
            if sequences:
                avg_sequence_length = sum(len(seq) for seq in sequences) / len(sequences)
                temporal_metrics['social_engagement_persistence'] = avg_sequence_length / SAMPLING_RATE  # Convert to seconds
                temporal_metrics['social_engagement_sequences'] = sequences
        
        return temporal_metrics
    
    def _analyze_fixation_patterns(self, df: pd.DataFrame) -> dict:
        """
        Analyze fixation patterns related to social attention.
        
        Args:
            df: Dataframe with ROI and fixation information
            
        Returns:
            dict: Fixation pattern metrics
        """
        fixation_metrics = {
            'social_fixation_count': 0,
            'nonsocial_fixation_count': 0,
            'social_fixation_duration_avg': 0,
            'nonsocial_fixation_duration_avg': 0,
            'fixation_duration_by_roi': {}
        }
        
        # Ensure we have fixation and ROI data
        required_cols = ['is_fixation_left', 'is_fixation_right', 'is_social_roi', 'roi_label']
        if not all(col in df.columns for col in required_cols):
            return fixation_metrics
        
        # Use both eyes' fixation data
        df['is_fixation'] = df['is_fixation_left'] & df['is_fixation_right']
        
        # Find continuous fixation blocks
        in_fixation = False
        fixation_start = 0
        fixation_blocks = []
        
        for idx, row in df.iterrows():
            if row['is_fixation'] and not in_fixation:
                # Start new fixation
                in_fixation = True
                fixation_start = idx
                current_roi = row['roi_label']
                is_social = row['is_social_roi']
            elif not row['is_fixation'] and in_fixation:
                # End fixation
                in_fixation = False
                fixation_blocks.append({
                    'start': fixation_start,
                    'end': idx - 1,
                    'duration': (idx - fixation_start) / SAMPLING_RATE,  # seconds
                    'roi': current_roi,
                    'is_social': is_social
                })
            elif in_fixation and (row['roi_label'] != current_roi or row['is_social_roi'] != is_social):
                # Changed ROI during fixation - end previous and start new
                fixation_blocks.append({
                    'start': fixation_start,
                    'end': idx - 1,
                    'duration': (idx - fixation_start) / SAMPLING_RATE,  # seconds
                    'roi': current_roi,
                    'is_social': is_social
                })
                fixation_start = idx
                current_roi = row['roi_label']
                is_social = row['is_social_roi']
        
        # Add last fixation if still active
        if in_fixation:
            fixation_blocks.append({
                'start': fixation_start,
                'end': len(df) - 1,
                'duration': (len(df) - fixation_start) / SAMPLING_RATE,  # seconds
                'roi': current_roi,
                'is_social': is_social
            })
        
        # Calculate fixation metrics
        social_fixations = [f for f in fixation_blocks if f['is_social']]
        nonsocial_fixations = [f for f in fixation_blocks if not f['is_social']]
        
        fixation_metrics['social_fixation_count'] = len(social_fixations)
        fixation_metrics['nonsocial_fixation_count'] = len(nonsocial_fixations)
        
        if social_fixations:
            fixation_metrics['social_fixation_duration_avg'] = sum(f['duration'] for f in social_fixations) / len(social_fixations)
        
        if nonsocial_fixations:
            fixation_metrics['nonsocial_fixation_duration_avg'] = sum(f['duration'] for f in nonsocial_fixations) / len(nonsocial_fixations)
        
        # Calculate fixation duration by ROI
        roi_fixation_durations = {}
        for fixation in fixation_blocks:
            roi = fixation['roi']
            if roi not in roi_fixation_durations:
                roi_fixation_durations[roi] = []
            roi_fixation_durations[roi].append(fixation['duration'])
        
        # Calculate average duration by ROI
        for roi, durations in roi_fixation_durations.items():
            fixation_metrics['fixation_duration_by_roi'][roi] = sum(durations) / len(durations)
        
        return fixation_metrics
    
    def extract_social_attention_features(self, metrics: dict) -> dict:
        """
        Extract social attention features for machine learning.
        
        Args:
            metrics: Social attention metrics dictionary
            
        Returns:
            dict: Dictionary of social attention features
        """
        features = {}
        
        # Core metrics
        features['social_attention_percent'] = metrics.get('social_attention_percent', 0)
        features['person_attention_percent'] = metrics.get('person_attention_percent', 0)
        features['face_attention_percent'] = metrics.get('face_attention_percent', 0)
        features['head_attention_percent'] = metrics.get('head_attention_percent', 0)
        features['hand_attention_percent'] = metrics.get('hand_attention_percent', 0)
        features['torso_attention_percent'] = metrics.get('torso_attention_percent', 0)
        features['object_attention_percent'] = metrics.get('object_attention_percent', 0)
        
        # Transition metrics
        transitions = metrics.get('transitions', {})
        features['social_to_nonsocial_transitions'] = transitions.get('social_to_nonsocial', 0)
        features['nonsocial_to_social_transitions'] = transitions.get('nonsocial_to_social', 0)
        features['transition_rate'] = transitions.get('total_transitions', 0) / metrics.get('total_valid_frames', 1)
        
        # First fixation
        features['first_fixation_social'] = 1 if metrics.get('first_fixation') in self.social_categories else 0
        features['time_to_first_social_look'] = metrics.get('time_to_first_social_look', 999)
        
        # Temporal patterns
        temporal = metrics.get('temporal_patterns', {})
        features['social_engagement_persistence'] = temporal.get('social_engagement_persistence', 0)
        
        # Fixation patterns
        fixation = metrics.get('fixation_patterns', {})
        features['social_fixation_count'] = fixation.get('social_fixation_count', 0)
        features['nonsocial_fixation_count'] = fixation.get('nonsocial_fixation_count', 0)
        features['social_fixation_ratio'] = features['social_fixation_count'] / max(1, features['social_fixation_count'] + features['nonsocial_fixation_count'])
        features['social_fixation_duration_avg'] = fixation.get('social_fixation_duration_avg', 0)
        features['nonsocial_fixation_duration_avg'] = fixation.get('nonsocial_fixation_duration_avg', 0)
        
        # Get specific ROI fixation durations
        roi_durations = fixation.get('fixation_duration_by_roi', {})
        for roi in ['Face', 'Head', 'Hand', 'Torso', 'Couch', 'Bed']:
            features[f'{roi.lower()}_fixation_duration_avg'] = roi_durations.get(roi, 0)
        
        return features


def extract_combined_features(df: pd.DataFrame, roi_analyzer: ROIAnalyzer) -> Dict[str, float]:
    """
    Extract both basic and social attention features.
    
    Args:
        df: Preprocessed dataframe
        roi_analyzer: ROIAnalyzer instance
        
    Returns:
        features: Dictionary of all features
    """
    # Extract basic features
    basic_features = extract_basic_features(df)
    
    # Extract social attention features
    social_attention_metrics = roi_analyzer.analyze_gaze_social_attention(df)
    social_features = roi_analyzer.extract_social_attention_features(social_attention_metrics)
    
    # Combine features
    combined_features = {**basic_features, **social_features}
    
    return combined_features