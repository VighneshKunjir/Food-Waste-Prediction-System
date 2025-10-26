# core/data_loader.py
"""Data loading utilities"""

import pandas as pd
import os


class DataLoader:
    """Handle data loading from various sources"""
    
    @staticmethod
    def load_csv(filepath):
        """Load data from CSV file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    @staticmethod
    def validate_data(df, required_columns=None):
        """Validate loaded data"""
        if df.empty:
            raise ValueError("Loaded dataframe is empty")
        
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        print(f"✅ Data validation passed")
        return True
    
    @staticmethod
    def preview_data(df, n=5):
        """Display data preview"""
        print(f"\n📊 Data Preview ({n} rows):")
        print(df.head(n))
        print(f"\n📋 Columns: {list(df.columns)}")
        print(f"📈 Shape: {df.shape}")
        
        return df.head(n)