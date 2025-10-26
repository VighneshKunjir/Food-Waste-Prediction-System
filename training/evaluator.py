# training/evaluator.py
"""Model evaluation with comprehensive metrics"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


class ModelEvaluator:
    """Evaluate models with comprehensive metrics"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Ensure numpy arrays
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, model_name)
        
        # Store results
        self.results[model_name] = metrics
        self.predictions[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        # Display metrics
        self._display_metrics(metrics)
        
        return metrics
    
    def evaluate_multiple_models(self, models_dict, X_test, y_test):
        """Evaluate multiple models"""
        print("\n" + "="*60)
        print("📊 EVALUATING MULTIPLE MODELS")
        print("="*60)
        
        all_metrics = []
        
        for model_name, model_data in models_dict.items():
            # Extract model from dict
            if isinstance(model_data, dict):
                model = model_data.get('model')
            else:
                model = model_data
            
            if model is None:
                print(f"⚠️ Skipping {model_name} - no model found")
                continue
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            all_metrics.append(metrics)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('MAE (kg)')
        
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*60)
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        best_mae = comparison_df.iloc[0]['MAE (kg)']
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   MAE: {best_mae:.3f} kg")
        print(f"   (Average prediction error: {best_mae:.2f} kg)")
        
        return comparison_df
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate all metrics"""
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handling zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0
        
        # Additional metrics
        median_ae = np.median(np.abs(y_true - y_pred))
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Accuracy within thresholds (kg-based)
        within_1kg = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
        within_2kg = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
        within_5kg = np.mean(np.abs(y_true - y_pred) <= 5.0) * 100
        
        # Over/Under prediction analysis
        over_predictions = np.mean(y_pred > y_true) * 100
        under_predictions = np.mean(y_pred < y_true) * 100
        
        metrics = {
            'Model': model_name,
            'MAE (kg)': mae,
            'RMSE (kg)': rmse,
            'R²': r2,
            'MAPE (%)': mape,
            'Median AE': median_ae,
            'Max Error': max_error,
            'Within 1kg (%)': within_1kg,
            'Within 2kg (%)': within_2kg,
            'Within 5kg (%)': within_5kg,
            'Over-predicted (%)': over_predictions,
            'Under-predicted (%)': under_predictions
        }
        
        return metrics
    
    def _display_metrics(self, metrics):
        """Display metrics in formatted way"""
        print(f"   MAE: {metrics['MAE (kg)']:.3f} kg")
        print(f"   RMSE: {metrics['RMSE (kg)']:.3f} kg")
        print(f"   R²: {metrics['R²']:.3f}")
        print(f"   Within 1kg: {metrics['Within 1kg (%)']:.1f}%")
        print(f"   Within 2kg: {metrics['Within 2kg (%)']:.1f}%")
        print(f"   Within 5kg: {metrics['Within 5kg (%)']:.1f}%")
    
    def get_predictions(self, model_name):
        """Get predictions for a specific model"""
        return self.predictions.get(model_name)
    
    def get_all_results(self):
        """Get all evaluation results"""
        return self.results
    
    def export_results(self, filepath):
        """Export results to CSV"""
        if not self.results:
            print("⚠️ No results to export")
            return
        
        df = pd.DataFrame(list(self.results.values()))
        df.to_csv(filepath, index=False)
        print(f"✅ Results exported to {filepath}")