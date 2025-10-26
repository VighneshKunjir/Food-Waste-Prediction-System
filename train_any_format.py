# train_any_format.py
"""
Universal Training Script - Works with ANY CSV Format
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime  # ADD THIS LINE
import json  # ADD THIS LINE TOO (if not already there)

from src.universal_data_adapter import UniversalDataAdapter
from src.restaurant_config_manager import RestaurantConfigManager
from src.data_preprocessing import DataPreprocessor
from src.model_training_gpu import ModelTrainerGPU
from src.model_training import ModelTrainer
import torch
import joblib

def train_with_your_format():
    """
    Specifically for your CSV format:
    Type of Food, Number of Guests, Event Type, Quantity of Food,
    Storage Conditions, Purchase History, Seasonality, Preparation Method,
    Geographical Location, Pricing, Wastage Food Amount
    """
    
    print("\n" + "="*60)
    print("🍔 UNIVERSAL FOOD WASTE PREDICTION TRAINING")
    print("="*60)
    
    # Get file path
    filepath = input("\n📁 Enter your CSV file path: ").strip().strip('"')
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    # Get restaurant name
    restaurant_name = input("🏪 Enter restaurant name (for saving config): ").strip()
    
    # Initialize adapter
    adapter = UniversalDataAdapter()
    config_manager = RestaurantConfigManager()
    
    # Check if we have a saved configuration
    saved_config = config_manager.load_restaurant_config(restaurant_name)
    
    if saved_config:
        use_saved = input("\n📋 Found saved configuration. Use it? (y/n): ").strip().lower()
        if use_saved == 'y':
            # Use saved mappings
            df = pd.read_csv(filepath)
            adapted_df = adapter.adapt_dataframe(df, saved_config['column_mappings'])
            adapted_df = adapter.infer_missing_columns(adapted_df)
        else:
            # Process with auto-detection
            adapted_df, mapping_result = adapter.process_csv(filepath)
            # Save new configuration
            config_manager.save_restaurant_config(restaurant_name, mapping_result)
    else:
        # Process with auto-detection
        adapted_df, mapping_result = adapter.process_csv(filepath)
        # Save configuration for future use
        config_manager.save_restaurant_config(restaurant_name, mapping_result)
    
    # Display adapted data info
    print("\n📊 ADAPTED DATA SUMMARY")
    print("="*50)
    print(f"Rows: {len(adapted_df)}")
    print(f"Columns: {adapted_df.columns.tolist()}")
    print("\nFirst few rows:")
    print(adapted_df.head())
    
    # Check if we have minimum required columns
    if 'food_item' not in adapted_df.columns or 'quantity_wasted' not in adapted_df.columns:
        print("\n❌ Critical columns missing. Cannot proceed with training.")
        print("Minimum required: food_item, quantity_wasted")
        return
    
    # Add any extra processing for special columns
    if 'extra_storage_conditions' in adapted_df.columns:
        print("\n🎯 Found extra feature: Storage Conditions")
        # Encode storage conditions
        storage_encoding = {
            'Refrigerated': 0, 'Frozen': 1, 'Room Temperature': 2, 
            'Chilled': 0, 'Ambient': 2, 'Cold': 0, 'Dry': 2
        }
        adapted_df['storage_encoded'] = adapted_df['extra_storage_conditions'].map(
            lambda x: storage_encoding.get(x, 2)
        )
    
    if 'extra_preparation_method' in adapted_df.columns:
        print("🎯 Found extra feature: Preparation Method")
        # Encode preparation methods
        prep_encoding = {
            'Grilled': 0, 'Fried': 1, 'Baked': 2, 'Steamed': 3,
            'Raw': 4, 'Boiled': 5, 'Roasted': 2
        }
        adapted_df['prep_encoded'] = adapted_df['extra_preparation_method'].map(
            lambda x: prep_encoding.get(x, 0)
        )
    
    # Continue with training
    proceed = input("\n🚀 Proceed with training? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("Training cancelled.")
        return
    
    # Save adapted data
    adapted_df.to_csv('data/raw/training_data.csv', index=False)
    
    # Initialize preprocessor
    print("\n🔧 PREPROCESSING DATA")
    print("-"*40)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data('data/raw/training_data.csv')
    
    # Train models
    print("\n🤖 TRAINING MODELS")
    print("-"*40)
    
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = input("🔥 GPU available. Use GPU acceleration? (y/n): ").strip().lower() == 'y'
    
    if use_gpu:
        trainer = ModelTrainerGPU()
    else:
        trainer = ModelTrainer()
    
    results = trainer.train_all_models(X_train, y_train)
    
    # Save models with restaurant name
    os.makedirs(f'data/models/{restaurant_name}', exist_ok=True)
    
    model_path = f'data/models/{restaurant_name}/best_model.pkl'
    preprocessor_path = f'data/models/{restaurant_name}/preprocessor.pkl'
    
    joblib.dump(trainer.best_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"\n✅ Models saved for {restaurant_name}")
    print(f"  Model: {model_path}")
    print(f"  Preprocessor: {preprocessor_path}")
    
    # Save training summary
    summary = {
        'restaurant': restaurant_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_model': trainer.best_model_name,
        'mae': trainer.best_score,
        'original_columns': list(pd.read_csv(filepath).columns),
        'rows_trained': len(adapted_df)
    }
    
    with open(f'data/models/{restaurant_name}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"🏆 Best Model: {trainer.best_model_name}")
    print(f"📊 Accuracy (MAE): {trainer.best_score:.2f} kg")

if __name__ == "__main__":
    # Install required package for fuzzy matching
    try:
        from fuzzywuzzy import fuzz
    except:
        print("Installing required package...")
        os.system("pip install fuzzywuzzy python-Levenshtein")
    
    train_with_your_format()