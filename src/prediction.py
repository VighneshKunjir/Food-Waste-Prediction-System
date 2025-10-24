# src/prediction.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WastePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.preprocessor = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        print(f"📂 Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
    def load_preprocessor(self, preprocessor):
        """Load the preprocessor used during training"""
        self.preprocessor = preprocessor
        
    def prepare_input_data(self, input_data):
        """Prepare input data for prediction"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Use the prediction-specific preprocessing
        if self.preprocessor:
            df_scaled = self.preprocessor.prepare_prediction_data(df)
            return df_scaled
        
        return df
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Prepare input data
        X = self.prepare_input_data(input_data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_confidence(self, input_data, confidence_level=0.95):
        """Make predictions with confidence intervals"""
        predictions = self.predict(input_data)
        
        # For ensemble or tree-based models
        if hasattr(self.model, 'estimators_'):
            X = self.prepare_input_data(input_data)
            
            # Get predictions from each estimator
            if hasattr(self.model, 'estimators_'):
                all_predictions = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'predict'):
                        try:
                            pred = estimator.predict(X)
                            all_predictions.append(pred)
                        except:
                            pass
                
                if all_predictions:
                    all_predictions = np.array(all_predictions)
                    std_dev = np.std(all_predictions, axis=0)
                    
                    # Calculate confidence intervals
                    z_score = 1.96  # for 95% confidence
                    lower_bound = predictions - z_score * std_dev
                    upper_bound = predictions + z_score * std_dev
                else:
                    # Fallback to simple confidence interval
                    margin = predictions * 0.15
                    lower_bound = predictions - margin
                    upper_bound = predictions + margin
            else:
                # Simple confidence interval
                margin = predictions * 0.15
                lower_bound = predictions - margin
                upper_bound = predictions + margin
        else:
            # For other models, use a simple percentage-based confidence
            margin = predictions * 0.15
            lower_bound = predictions - margin
            upper_bound = predictions + margin
        
        return {
            'predictions': predictions,
            'lower_bound': np.maximum(lower_bound, 0),
            'upper_bound': upper_bound
        }
    
    def batch_predict(self, csv_path):
        """Make predictions on a batch of data from CSV"""
        print(f"\n📂 Loading batch data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"🔮 Making predictions for {len(df)} records...")
        predictions = self.predict(df)
        
        # Add predictions to dataframe
        df['predicted_waste'] = predictions
        
        # Calculate waste reduction potential if actual waste is available
        if 'quantity_wasted' in df.columns:
            df['waste_reduction_potential'] = df['quantity_wasted'] - df['predicted_waste']
            df['potential_savings'] = df['waste_reduction_potential'] * df.get('unit_cost', 5)
        
        # Save results
        output_path = csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"💾 Predictions saved to {output_path}")
        
        # Display summary
        print("\n📊 Prediction Summary:")
        print(f"   Total predicted waste: {predictions.sum():.2f}")
        print(f"   Average predicted waste: {predictions.mean():.2f}")
        if 'potential_savings' in df.columns:
            print(f"   Potential total savings: ${df['potential_savings'].sum():,.2f}")
        
        return df
    
    def real_time_prediction(self):
        """Interactive real-time prediction"""
        print("\n🔮 REAL-TIME WASTE PREDICTION")
        print("="*50)
        print("Enter 'quit' to exit\n")
        
        while True:
            try:
                # Get input from user
                print("\nEnter prediction details:")
                
                food_item = input("Food item: ")
                if food_item.lower() == 'quit':
                    break
                    
                input_data = {
                    'date': input("Date (YYYY-MM-DD): "),
                    'food_item': food_item,
                    'food_category': input("Food category (mains/sides/desserts/etc): "),
                    'quantity_prepared': float(input("Quantity prepared: ")),
                    'quantity_sold': float(input("Expected quantity sold: ")),
                    'day_of_week': input("Day of week: "),
                    'is_weekend': int(input("Is weekend? (0/1): ")),
                    'weather': input("Weather (sunny/cloudy/rainy/stormy): "),
                    'temperature': float(input("Temperature (°C): ")),
                    'special_event': input("Special event (none/wedding/birthday/etc): "),
                    'customer_count': int(input("Expected customers: ")),
                    'month': int(input("Month (1-12): ")),
                    'unit_cost': float(input("Unit cost ($): "))
                }
                
                # Make prediction
                result = self.predict_with_confidence(input_data)
                
                print("\n" + "="*50)
                print("📊 PREDICTION RESULTS")
                print("="*50)
                print(f"Predicted waste: {result['predictions'][0]:.2f} units")
                print(f"Confidence interval: [{result['lower_bound'][0]:.2f}, {result['upper_bound'][0]:.2f}]")
                print(f"Estimated waste cost: ${result['predictions'][0] * input_data['unit_cost']:.2f}")
                
                # Recommendations
                waste_percentage = (result['predictions'][0] / input_data['quantity_prepared']) * 100
                
                print("\n💡 Recommendations:")
                if waste_percentage > 20:
                    print("   ⚠️ High waste predicted! Consider reducing preparation quantity.")
                elif waste_percentage > 10:
                    print("   ⚡ Moderate waste expected. Monitor closely.")
                else:
                    print("   ✅ Waste levels within acceptable range.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again with valid inputs.")