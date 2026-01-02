import pandas as pd
import os
from typing import Dict

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Loads a CSV file from the data directory. Handles fully quoted lines."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path} - Current working directory: {os.getcwd()}")
        
        # Pre-process to handle files where entire lines are quoted
        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            # Fix double quotes being escaped as "" due to full line quoting
            line = line.replace('""', '"')
            if line:
                cleaned_lines.append(line)
                
        from io import StringIO
        return pd.read_csv(StringIO('\n'.join(cleaned_lines)), sep=',', skipinitialspace=True)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Loads all required datasets."""
        try:
            users = self.load_csv("users.csv")
            events = self.load_csv("events.csv")
            rules = self.load_csv("guidance_rules.csv")
            return {
                "users": users,
                "events": events,
                "rules": rules
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

def clean_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Performs basic cleaning on the datasets."""
    if "users" in data:
        data["users"]["user_id"] = data["users"]["user_id"].astype(int)
    
    if "events" in data:
        data["events"]["user_id"] = data["events"]["user_id"].astype(int)
        
    # Strip whitespace from string columns
    for df_name, df in data.items():
        if df is not None:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            data[df_name] = df
        
    return data
