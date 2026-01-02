import argparse
import json
import sys
import os

# Add src to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader, clean_data
from src.model_pipeline import GuidanceEngine

def main():
    parser = argparse.ArgumentParser(description="AI Life Guidance System")
    parser.add_argument("--user_id", type=int, help="User ID to generate guidance for", required=True)
    args = parser.parse_args()

    # Load Data
    try:
        loader = DataLoader() # Defaults to 'data' dir relative to CWD
        raw_data = loader.load_all()
        data = clean_data(raw_data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return

    # Initialize Engine
    engine = GuidanceEngine(data)
    
    # Train 'Model' (In a real app, we'd load a saved model)
    engine.train_stress_model()
    
    # Get Guidance
    result = engine.get_guidance(args.user_id)
    
    # Output
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
