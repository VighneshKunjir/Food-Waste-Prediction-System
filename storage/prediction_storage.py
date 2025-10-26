# storage/prediction_storage.py
"""Prediction storage and history management"""

import os
import json
import pandas as pd
from datetime import datetime
from utils.file_utils import FileManager


class PredictionStorage:
    """Manage prediction storage and history"""
    
    def __init__(self, base_dir='data/restaurants'):
        self.base_dir = base_dir
    
    def save_prediction(self, restaurant_name, prediction_data, prediction_type='realtime'):
        """
        Save prediction with metadata
        
        Args:
            restaurant_name: Name of restaurant
            prediction_data: Dictionary or DataFrame with prediction data
            prediction_type: 'realtime' or 'future'
        
        Returns:
            Path to saved prediction file
        """
        # Create directory
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        predictions_dir = os.path.join(restaurant_dir, 'predictions', prediction_type)
        FileManager.create_directory(predictions_dir)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"prediction_{timestamp}.json"
        filepath = os.path.join(predictions_dir, filename)
        
        # Prepare data
        if isinstance(prediction_data, pd.DataFrame):
            # Convert DataFrame to dict
            data = {
                'timestamp': timestamp,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_type': prediction_type,
                'data': prediction_data.to_dict('records')
            }
        else:
            # Already a dict
            data = {
                'timestamp': timestamp,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_type': prediction_type,
                'data': prediction_data
            }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"💾 Prediction saved: {filepath}")
        
        # Also save as CSV if DataFrame
        if isinstance(prediction_data, pd.DataFrame):
            csv_filepath = filepath.replace('.json', '.csv')
            prediction_data.to_csv(csv_filepath, index=False)
            print(f"💾 CSV saved: {csv_filepath}")
        
        return filepath
    
    def load_prediction(self, restaurant_name, timestamp, prediction_type='realtime'):
        """Load a specific prediction"""
        predictions_dir = os.path.join(self.base_dir, restaurant_name, 'predictions', prediction_type)
        filepath = os.path.join(predictions_dir, f"prediction_{timestamp}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prediction not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def get_prediction_history(self, restaurant_name, prediction_type='realtime', limit=10):
        """Get recent prediction history"""
        predictions_dir = os.path.join(self.base_dir, restaurant_name, 'predictions', prediction_type)
        
        if not os.path.exists(predictions_dir):
            return []
        
        # Get all prediction files
        prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.json')]
        prediction_files.sort(reverse=True)  # Newest first
        
        # Load recent predictions
        history = []
        for filename in prediction_files[:limit]:
            filepath = os.path.join(predictions_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                history.append(data)
            except:
                pass
        
        return history
    
    def get_all_predictions_df(self, restaurant_name, prediction_type='realtime'):
        """Get all predictions as a single DataFrame"""
        predictions_dir = os.path.join(self.base_dir, restaurant_name, 'predictions', prediction_type)
        
        if not os.path.exists(predictions_dir):
            return pd.DataFrame()
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
        
        if not csv_files:
            return pd.DataFrame()
        
        # Load and concatenate
        dfs = []
        for csv_file in csv_files:
            filepath = os.path.join(predictions_dir, csv_file)
            try:
                df = pd.read_csv(filepath)
                dfs.append(df)
            except:
                pass
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def delete_prediction(self, restaurant_name, timestamp, prediction_type='realtime'):
        """Delete a specific prediction"""
        predictions_dir = os.path.join(self.base_dir, restaurant_name, 'predictions', prediction_type)
        
        json_path = os.path.join(predictions_dir, f"prediction_{timestamp}.json")
        csv_path = os.path.join(predictions_dir, f"prediction_{timestamp}.csv")
        
        deleted = False
        
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"🗑️ Deleted: {json_path}")
            deleted = True
        
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"🗑️ Deleted: {csv_path}")
        
        return deleted
    
    def clean_old_predictions(self, restaurant_name, days=30, prediction_type='realtime'):
        """Delete predictions older than specified days"""
        predictions_dir = os.path.join(self.base_dir, restaurant_name, 'predictions', prediction_type)
        
        if not os.path.exists(predictions_dir):
            return 0
        
        deleted_count = FileManager.clean_old_files(predictions_dir, days=days, extension='.json')
        FileManager.clean_old_files(predictions_dir, days=days, extension='.csv')
        
        return deleted_count
    
    def get_prediction_summary(self, restaurant_name):
        """Get summary of all predictions for a restaurant"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        predictions_dir = os.path.join(restaurant_dir, 'predictions')
        
        if not os.path.exists(predictions_dir):
            return {
                'restaurant': restaurant_name,
                'total_predictions': 0,
                'realtime_predictions': 0,
                'future_predictions': 0
            }
        
        realtime_dir = os.path.join(predictions_dir, 'realtime')
        future_dir = os.path.join(predictions_dir, 'future')
        
        realtime_count = len([f for f in os.listdir(realtime_dir) if f.endswith('.json')]) if os.path.exists(realtime_dir) else 0
        future_count = len([f for f in os.listdir(future_dir) if f.endswith('.json')]) if os.path.exists(future_dir) else 0
        
        summary = {
            'restaurant': restaurant_name,
            'total_predictions': realtime_count + future_count,
            'realtime_predictions': realtime_count,
            'future_predictions': future_count,
            'predictions_directory': predictions_dir
        }
        
        return summary
    
    def export_predictions_report(self, restaurant_name, output_path=None):
        """Export comprehensive predictions report"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"results/reports/{restaurant_name}_predictions_report_{timestamp}.csv"
        
        # Get all predictions
        realtime_df = self.get_all_predictions_df(restaurant_name, 'realtime')
        future_df = self.get_all_predictions_df(restaurant_name, 'future')
        
        if realtime_df.empty and future_df.empty:
            print("⚠️ No predictions found to export")
            return None
        
        # Add prediction type column
        if not realtime_df.empty:
            realtime_df['prediction_type'] = 'realtime'
        if not future_df.empty:
            future_df['prediction_type'] = 'future'
        
        # Combine
        if not realtime_df.empty and not future_df.empty:
            combined_df = pd.concat([realtime_df, future_df], ignore_index=True)
        elif not realtime_df.empty:
            combined_df = realtime_df
        else:
            combined_df = future_df
        
        # Save report
        FileManager.save_csv(combined_df, output_path)
        print(f"📊 Predictions report exported: {output_path}")
        
        return output_path