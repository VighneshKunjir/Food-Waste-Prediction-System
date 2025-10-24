# predict_future.py
"""
Standalone script for future waste predictions
Can be run independently after model training
"""

import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.future_prediction import FutureWastePredictor

def main():
    print("\n" + "="*60)
    print("🔮 FOOD WASTE FUTURE PREDICTION SYSTEM")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists('data/models/best_waste_model.pkl'):
        print("\n⚠️ No trained model found!")
        print("Please run 'python main_gpu.py' first to train the model.")
        return
    
    # Initialize predictor
    predictor = FutureWastePredictor()
    
    # Run interactive predictions
    predictor.run_interactive_prediction()

if __name__ == "__main__":
    main()