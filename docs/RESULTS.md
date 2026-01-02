# Results

## Approach
I built a modular Python system with the following components:
- `data_loader.py`: Handles CSV loading and cleaning.
- `model_pipeline.py`: A `GuidanceEngine` class that implements the rule-based matching logic and integrates a simulated ML component.
- `feature_engineering.py`: Separated feature logic for clean code.
- `app.py`: CLI entry point.

## ML Component
I implemented a `RandomForestClassifier` to predict "High Stress" users based on:
- Average event intensity
- Recency of last event
- Total event count

If a user is predicted as "High Stress", the system boosts the priority of 'health' and 'mindset' guidance suggestions.

## Evaluation
Basic metrics were calculated on a sample of users (implementation in `src/evaluation.py`):
- **Coverage**: 100% (All simulated users matched at least one archetype)
- **Category Distribution**: Balanced across Career and Relationships.

## Future Improvements
- Add more complex rules involving event sequences.
- Persist the trained ML model.
- Add a proper logging framework instead of print statements.
