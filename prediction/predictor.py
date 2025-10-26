# prediction/predictor.py
"""Base predictor class"""

import numpy as np
import pandas as pd
import joblib


class BasePredictor:
    """Base class for all predictors"""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        self.model = None
        self.preprocessor = None
        
        if model_path:
            self.load_model(model_path)
        
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        print(f"📂 Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    
    def load_preprocessor(self, preprocessor_path):
        """Load preprocessor"""
        print(f"📂 Loading preprocessor from {preprocessor_path}...")
        self.preprocessor = joblib.load(preprocessor_path)
        print("✅ Preprocessor loaded successfully")
    
    def prepare_input(self, data):
        """Prepare input data for prediction"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Preprocess
        data_processed = self.preprocessor.transform(data)
        
        return data_processed
    
    def predict(self, data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare input
        X = self.prepare_input(data)
        
        # Convert DataFrame to numpy array to avoid sklearn warnings
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        
        # Predict
        predictions = self.model.predict(X_array)
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_confidence(self, data, confidence_level=0.95):
        """Make predictions with confidence intervals"""
        predictions = self.predict(data)
        
        # Calculate confidence intervals
        if hasattr(self.model, 'estimators_'):
            # For ensemble models, use estimator variance
            X = self.prepare_input(data)
            
            # Convert to numpy
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X
            
            all_predictions = []
            
            for estimator in self.model.estimators_:
                try:
                    pred = estimator.predict(X_array)
                    all_predictions.append(pred)
                except:
                    pass
            
            if all_predictions:
                all_predictions = np.array(all_predictions)
                std_dev = np.std(all_predictions, axis=0)
                
                # 95% confidence interval (z-score = 1.96)
                z_score = 1.96
                lower_bound = predictions - z_score * std_dev
                upper_bound = predictions + z_score * std_dev
            else:
                # Fallback: 15% margin
                margin = predictions * 0.15
                lower_bound = predictions - margin
                upper_bound = predictions + margin
        else:
            # Simple confidence interval (15% margin)
            margin = predictions * 0.15
            lower_bound = predictions - margin
            upper_bound = predictions + margin
        
        return {
            'predictions': predictions,
            'lower_bound': np.maximum(lower_bound, 0),
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }
    
    def batch_predict(self, data_path, output_path=None):
        """Make predictions on batch data from CSV"""
        print(f"\n📂 Loading batch data from {data_path}...")
        df = pd.read_csv(data_path)
        
        print(f"🔮 Making predictions for {len(df)} records...")
        predictions = self.predict(df)
        
        # Add predictions to dataframe
        df['predicted_waste'] = predictions
        
        # Calculate potential savings if actual waste exists
        if 'quantity_wasted' in df.columns:
            df['prediction_error'] = df['quantity_wasted'] - df['predicted_waste']
            df['absolute_error'] = np.abs(df['prediction_error'])
            
            if 'unit_cost' in df.columns:
                df['waste_cost_actual'] = df['quantity_wasted'] * df['unit_cost']
                df['waste_cost_predicted'] = df['predicted_waste'] * df['unit_cost']
                df['potential_savings'] = df['waste_cost_actual'] - df['waste_cost_predicted']
        
        # Save results
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to {output_path}")
        
        # Display summary
        self._display_batch_summary(df, predictions)
        
        return df
    
    def _display_batch_summary(self, df, predictions):
        """Display batch prediction summary"""
        print("\n📊 BATCH PREDICTION SUMMARY")
        print("="*50)
        print(f"   Total records: {len(df)}")
        print(f"   Total predicted waste: {predictions.sum():.2f} kg")
        print(f"   Average predicted waste: {predictions.mean():.2f} kg")
        print(f"   Min predicted waste: {predictions.min():.2f} kg")
        print(f"   Max predicted waste: {predictions.max():.2f} kg")
        
        if 'quantity_wasted' in df.columns:
            mae = np.mean(np.abs(df['quantity_wasted'] - predictions))
            print(f"\n   Prediction Accuracy:")
            print(f"   MAE: {mae:.2f} kg")
            
            if 'potential_savings' in df.columns:
                total_savings = df['potential_savings'].sum()
                print(f"\n   Financial Impact:")
                print(f"   Potential savings: ${total_savings:,.2f}")