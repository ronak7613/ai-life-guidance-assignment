import pandas as pd
import numpy as np
from typing import Tuple

def extract_stress_features(events_df: pd.DataFrame, user_ids: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features for the stress prediction model.
    Returns X (features) and y (synthetic labels).
    """
    features = []
    labels = []
    
    for user_id in user_ids:
        u_events = events_df[events_df['user_id'] == user_id]
        if u_events.empty:
            continue
            
        avg_intensity = u_events['event_intensity'].mean()
        min_recency = u_events['event_recency_days'].min()
        event_count = len(u_events)
        
        # synthetic label: High Stress if avg intensity > 3.0
        is_stressed = 1 if avg_intensity > 3.0 else 0
        
        features.append([avg_intensity, min_recency, event_count])
        labels.append(is_stressed)
        
    return np.array(features), np.array(labels)

def predict_features_for_user(events_df: pd.DataFrame, user_id: int) -> np.ndarray:
    """Prepare single user features for inference"""
    u_events = events_df[events_df['user_id'] == user_id]
    if u_events.empty:
        return None
        
    avg_intensity = u_events['event_intensity'].mean()
    min_recency = u_events['event_recency_days'].min()
    event_count = len(u_events)
    
    return np.array([[avg_intensity, min_recency, event_count]])
