import pytest
import pandas as pd
from src.app import main
from src.model_pipeline import GuidanceEngine

# Mock data for integration test
@pytest.fixture
def integration_data():
    users = pd.DataFrame([{
        'user_id': 999, 'profession': 'Test_Subject', 
        'personality_type': 'Stoic', 'risk_appetite': 'Low'
    }])
    events = pd.DataFrame([
        {'event_id': 1, 'user_id': 999, 'event_type': 'Test_Event', 'event_intensity': 2, 'event_recency_days': 1}
    ])
    rules = pd.DataFrame([
        {
            'rule_id': 'T1', 'archetype': 'Tester', 
            'condition_type': 'profession', 'condition_value': 'Test_Subject',
            'recommended_category': 'test_cat', 'priority': 1, 'template_message': 'Test Msg'
        }
    ])
    return {'users': users, 'events': events, 'rules': rules}

def test_full_pipeline_flow(integration_data):
    """
    Tests the engine as it would be used in the app:
    Data -> Engine -> Model Training -> Prediction -> Guidance
    """
    engine = GuidanceEngine(integration_data)
    
    # 1. Train Model (should handle small data gracefully)
    engine.train_stress_model()
    
    # 2. Get Guidance
    result = engine.get_guidance(999)
    
    # 3. Assertions
    assert result['user_id'] == 999
    assert len(result['guidance']) == 1
    assert result['guidance'][0]['category'] == 'test_cat'
    assert "High Stress Detected" not in result['guidance'][0]['message'] # Intensity 2 < 3

def test_high_stress_integration():
    """Test that stress boosting works in full flow"""
    data = {
        'users': pd.DataFrame([{'user_id': 1}]),
        'events': pd.DataFrame([{'user_id': 1, 'event_intensity': 5, 'event_recency_days': 1}]),
        'rules': pd.DataFrame([{
            'condition_type': 'event_type', 'condition_value': 'ALL', # Dummy
            'recommended_category': 'health', 'priority': 3, 'template_message': 'Health Msg',
            'archetype': 'Stressed',
            # Hack to ensure rule matches: we need a matching condition. 
            # In real engine, we match strict values. Let's make a real match.
        }])
    }
    # Update rules to actually match something
    data['rules']['condition_type'] = 'user_id' # Hack: engine doesn't support user_id directly in logic provided?
    # Actually, looking at match_rules in model_pipeline.py:
    # if condition_type in user_profile.index: match user attribute
    
    data['users']['is_test'] = 'yes'
    data['rules']['condition_type'] = 'is_test'
    data['rules']['condition_value'] = 'yes'
    
    engine = GuidanceEngine(data)
    engine.train_stress_model()
    
    result = engine.get_guidance(1)
    
    # Priority should be boosted from 3 to 4 because Intensity 5 > 3 -> Stressed
    assert result['guidance'][0]['priority'] == 4
    assert "[High Stress Detected]" in result['guidance'][0]['message']
