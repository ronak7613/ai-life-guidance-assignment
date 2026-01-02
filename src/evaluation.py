import pandas as pd
import numpy as np
from typing import Dict, List
from src.model_pipeline import GuidanceEngine

def evaluate_system(engine: GuidanceEngine, sample_size: int = 10) -> Dict[str, Any]:
    """
    Runs basic evaluation on the system.
    Metrics:
    - Coverage: % of users receiving at least one suggestion
    - Category Distribution
    - Stress Model Label Balance
    """
    user_ids = engine.users['user_id'].unique()
    sample_users = user_ids[:sample_size]
    
    results = []
    category_counts = {}
    total_recs = 0
    users_with_recs = 0
    
    for uid in sample_users:
        res = engine.get_guidance(uid)
        recs = res.get('guidance', [])
        
        if recs:
            users_with_recs += 1
            
        for rec in recs:
            cat = rec['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_recs += 1
            
    metrics = {
        "users_evaluated": len(sample_users),
        "coverage_percent": (users_with_recs / len(sample_users)) * 100,
        "avg_recs_per_user": total_recs / len(sample_users),
        "category_distribution": category_counts
    }
    
    return metrics
