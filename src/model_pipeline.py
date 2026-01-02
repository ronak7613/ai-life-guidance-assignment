from src.feature_engineering import extract_stress_features, predict_features_for_user
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier

class GuidanceEngine:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.users = data["users"]
        self.events = data["events"]
        self.rules = data["rules"]
        self.ml_model = None
        
    def _get_user_events(self, user_id: int) -> pd.DataFrame:
        return self.events[self.events['user_id'] == user_id]

    def _get_user_profile(self, user_id: int) -> pd.Series:
        user = self.users[self.users['user_id'] == user_id]
        if user.empty:
            raise ValueError(f"User ID {user_id} not found")
        return user.iloc[0]

    def match_rules(self, user_id: int) -> List[Dict[str, Any]]:
        user_profile = self._get_user_profile(user_id)
        user_events = self._get_user_events(user_id)
        
        recommendations = []
        
        # Iterate through rules
        for _, rule in self.rules.iterrows():
            condition_type = rule['condition_type']
            condition_value = rule['condition_value']
            
            match = False
            
            # Check Profile Attributes
            if condition_type in user_profile.index:
                if str(user_profile[condition_type]) == str(condition_value):
                    match = True
            
            # Check Event Attributes
            elif condition_type == 'event_type':
                # Check if user has any event of this type
                if not user_events.empty and condition_value in user_events['event_type'].values:
                    match = True
            
            if match:
                rec = {
                    "category": rule['recommended_category'],
                    "priority": int(rule['priority']),
                    "message": rule['template_message'],
                    "archetype": rule['archetype']
                }
                recommendations.append(rec)
                
        return recommendations

    def train_stress_model(self):
        """
        Trains a simple ML model to predict 'High Stress' based on event intensity and recency.
        """
        user_ids = self.users['user_id'].unique()
        X, y = extract_stress_features(self.events, user_ids)
            
        if len(X) == 0:
            print("Not enough data to train model.")
            return

        self.ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.ml_model.fit(X, y)

    def predict_stress(self, user_id: int) -> bool:
        if self.ml_model is None:
            return False
            
        features = predict_features_for_user(self.events, user_id)
        if features is None:
            return False
        
        prediction = self.ml_model.predict(features)
        return bool(prediction[0])

    def get_guidance(self, user_id: int) -> Dict[str, Any]:
        """
        Main entry point to get guidance for a user.
        Combines rule-based and ML-based insights.
        """
        try:
            base_recs = self.match_rules(user_id)
            is_stressed = self.predict_stress(user_id)
            
            # ML Personalization: Boost Health/Mindset priority if stressed
            if is_stressed:
                for rec in base_recs:
                    if rec['category'] in ['health', 'mindset']:
                        rec['priority'] += 1
                        rec['message'] = "[High Stress Detected] " + rec['message']
                
                # Fallback if no health guidance found matching rules? 
                # For clean code, we stick to rules for now, but could add a default message.
            
            # Sort by priority
            base_recs.sort(key=lambda x: x['priority'], reverse=True)
            
            return {
                "user_id": user_id,
                "guidance": base_recs[:4] # Return top 4
            }
        except ValueError as e:
            return {"error": str(e)}
