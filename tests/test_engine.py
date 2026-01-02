import pytest
import pandas as pd
from src.model_pipeline import GuidanceEngine

@pytest.fixture
def mock_data():
    users = pd.DataFrame([{
        'user_id': 1,
        'profession': 'Engineer',
        'personality_type': 'Analytical'
    }])
    events = pd.DataFrame([
        {
            'event_id': 101, 'user_id': 1, 'event_type': 'Work_Stress',
            'event_intensity': 5, 'event_recency_days': 2
        }
    ])
    rules = pd.DataFrame([
        {
            'rule_id': 'R1', 'archetype': 'Worker', 
            'condition_type': 'profession', 'condition_value': 'Engineer',
            'recommended_category': 'career', 'priority': 3, 'template_message': 'Work hard'
        }
    ])
    return {'users': users, 'events': events, 'rules': rules}

def test_engine_initialization(mock_data):
    engine = GuidanceEngine(mock_data)
    assert engine.users.shape[0] == 1

def test_rule_matching(mock_data):
    engine = GuidanceEngine(mock_data)
    recs = engine.match_rules(1)
    assert len(recs) == 1
    assert recs[0]['archetype'] == 'Worker'

def test_stress_prediction_logic(mock_data):
    engine = GuidanceEngine(mock_data)
    # Train heavily biased model on single sample
    engine.train_stress_model()
    # Predict
    is_stressed = engine.predict_stress(1)
    # Intensity 5 > 3.0 -> Should be True
    assert is_stressed is True
