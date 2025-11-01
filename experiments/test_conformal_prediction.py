# experiments/test_conformal_prediction.py
"""
Quick test script for Conformal Prediction with Rich Dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Our modules
from core.rich_data_generator import RichDataGenerator
from core.preprocessor import Preprocessor
from prediction.conformal_predictor import (
    ConformalPredictor, 
    train_conformal_predictor, 
    compare_prediction_methods
)
from utils.conformal_visualization import (
    plot_prediction_intervals, 
    plot_coverage_analysis, 
    plot_method_comparison,
    plot_conformal_dashboard  # NEW
)


def main():
    print("\n" + "="*70)
    print("🧪 CONFORMAL PREDICTION TEST")
    print("="*70)
    
    # Step 1: Generate Rich Dataset
    print("\n📊 Step 1: Generating Rich Dataset...")
    generator = RichDataGenerator(seed=42)
    
    df = generator.generate(
        n_days=365,  # 1 year
        restaurant_type='casual_dining',
        save_path='data/research/test_restaurant/data.csv'
    )
    
    # Step 2: Preprocess
    print("\n🔄 Step 2: Preprocessing...")
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Step 3: Train Conformal Predictor
    print("\n🎯 Step 3: Training Conformal Predictor...")
    
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    cp, X_proper, y_proper = train_conformal_predictor(
        model=base_model,
        X_train=X_train,
        y_train=y_train,
        cal_fraction=0.2,
        alpha=0.1,  # 90% coverage
        method='absolute'
    )
    
    # Step 4: Evaluate
    print("\n📈 Step 4: Evaluating on Test Set...")
    stats = cp.evaluate_coverage(X_test, y_test, verbose=True)
    
    # Step 5: Visualize
    print("\n📊 Step 5: Generating Visualizations...")
    os.makedirs('results/conformal_prediction', exist_ok=True)
    
    # Get predictions with intervals
    result = cp.predict_with_intervals(X_test)
    
    # Plot 1: Comprehensive Dashboard (NEW!)
    plot_conformal_dashboard(
        y_test, 
        result['predictions'],
        result['lower_bound'],
        result['upper_bound'],
        target_coverage=0.9,
        save_path='results/conformal_prediction/conformal_dashboard.png'
    )
    
    # Plot 2: Prediction intervals
    plot_prediction_intervals(
        y_test, 
        result['predictions'],
        result['lower_bound'],
        result['upper_bound'],
        n_samples=100,
        save_path='results/conformal_prediction/prediction_intervals.png'
    )
    
    # Plot 3: Coverage analysis
    plot_coverage_analysis(
        y_test,
        result['lower_bound'],
        result['upper_bound'],
        result['predictions'],
        target_coverage=0.9,
        save_path='results/conformal_prediction/coverage_analysis.png'
    )
    
    # Step 6: Compare Methods
    print("\n🔬 Step 6: Comparing Different Methods...")
    comparison_df = compare_prediction_methods(
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        alphas=[0.1, 0.05]
    )
    
    print("\n📊 Comparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plot_method_comparison(
        comparison_df,
        save_path='results/conformal_prediction/method_comparison.png'
    )
    
    # Step 7: Save Conformal Predictor
    print("\n💾 Step 7: Saving Conformal Predictor...")
    cp.save('data/research/test_restaurant/conformal_predictor.pkl')
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: results/conformal_prediction/")
    print(f"   - conformal_dashboard.png (NEW!)")
    print(f"   - prediction_intervals.png")
    print(f"   - coverage_analysis.png")
    print(f"   - method_comparison.png")
    print(f"\n💾 Model saved to: data/research/test_restaurant/conformal_predictor.pkl")
    
    print("\n📊 Summary:")
    print(f"   Target Coverage: {result['coverage_guarantee']*100:.0f}%")
    print(f"   Actual Coverage: {stats['actual_coverage']*100:.1f}%")
    print(f"   Status: {'✓ VALID' if stats['coverage_valid'] else '✗ INVALID'}")
    print(f"   Avg Interval Width: {stats['avg_interval_width']:.2f} kg")
    print(f"   MAE: {stats['mae']:.2f} kg")


if __name__ == "__main__":
    main()