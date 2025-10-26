# storage/model_storage.py
"""Model storage with metadata management"""

import os
import joblib
import json
from datetime import datetime
from utils.file_utils import FileManager


class ModelStorage:
    """Manage model storage and metadata"""
    
    def __init__(self, base_dir='data/restaurants'):
        self.base_dir = base_dir
        FileManager.create_directory(base_dir)
    
    def save_model(self, model, restaurant_name, metadata=None):
        """
        Save model with metadata
        
        Args:
            model: Trained model object
            restaurant_name: Name of restaurant
            metadata: Dictionary with model information
        
        Returns:
            Dictionary with file paths
        """
        # Create restaurant directory
        restaurant_dir = FileManager.get_restaurant_path(restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{timestamp}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_file': model_filename,
            'model_path': model_path
        })
        
        # Save metadata
        metadata_path = os.path.join(models_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"📋 Metadata saved: {metadata_path}")
        
        # Save as best model if specified
        if metadata.get('is_best', False):
            best_model_path = os.path.join(models_dir, 'best_model.pkl')
            joblib.dump(model, best_model_path)
            
            best_metadata_path = os.path.join(models_dir, 'best_model_metadata.json')
            with open(best_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"🏆 Saved as best model: {best_model_path}")
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'timestamp': timestamp
        }
    
    def save_preprocessor(self, preprocessor, restaurant_name):
        """Save preprocessor for a restaurant"""
        restaurant_dir = FileManager.get_restaurant_path(restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        joblib.dump(preprocessor, preprocessor_path)
        
        print(f"💾 Preprocessor saved: {preprocessor_path}")
        
        return preprocessor_path
    
    def load_model(self, restaurant_name, model_type='best'):
        """
        Load model for a restaurant
        
        Args:
            restaurant_name: Name of restaurant
            model_type: 'best' or timestamp of specific model
        
        Returns:
            Loaded model object
        """
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"No models found for restaurant: {restaurant_name}")
        
        if model_type == 'best':
            model_path = os.path.join(models_dir, 'best_model.pkl')
        else:
            model_path = os.path.join(models_dir, f"model_{model_type}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        return model
    
    def load_preprocessor(self, restaurant_name):
        """Load preprocessor for a restaurant"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor loaded: {preprocessor_path}")
        
        return preprocessor
    
    def load_metadata(self, restaurant_name, model_type='best'):
        """Load model metadata"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        if model_type == 'best':
            metadata_path = os.path.join(models_dir, 'best_model_metadata.json')
        else:
            metadata_path = os.path.join(models_dir, f"metadata_{model_type}.json")
        
        if not os.path.exists(metadata_path):
            print(f"⚠️ Metadata not found: {metadata_path}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self, restaurant_name):
        """List all saved models for a restaurant"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        if not os.path.exists(models_dir):
            return []
        
        model_files = [f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.pkl')]
        
        models_info = []
        for model_file in model_files:
            timestamp = model_file.replace('model_', '').replace('.pkl', '')
            metadata = self.load_metadata(restaurant_name, timestamp)
            
            models_info.append({
                'timestamp': timestamp,
                'filename': model_file,
                'metadata': metadata
            })
        
        # Sort by timestamp (newest first)
        models_info.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return models_info
    
    def delete_model(self, restaurant_name, model_timestamp):
        """Delete a specific model"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        model_path = os.path.join(models_dir, f"model_{model_timestamp}.pkl")
        metadata_path = os.path.join(models_dir, f"metadata_{model_timestamp}.json")
        
        deleted = False
        
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"🗑️ Deleted model: {model_path}")
            deleted = True
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"🗑️ Deleted metadata: {metadata_path}")
        
        return deleted
    
    def get_model_info(self, restaurant_name):
        """Get comprehensive model information"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        models_dir = os.path.join(restaurant_dir, 'models')
        
        if not os.path.exists(models_dir):
            return None
        
        info = {
            'restaurant': restaurant_name,
            'models_directory': models_dir,
            'has_best_model': os.path.exists(os.path.join(models_dir, 'best_model.pkl')),
            'has_preprocessor': os.path.exists(os.path.join(models_dir, 'preprocessor.pkl')),
            'total_models': len([f for f in os.listdir(models_dir) if f.startswith('model_')]),
            'best_model_metadata': self.load_metadata(restaurant_name, 'best') if os.path.exists(os.path.join(models_dir, 'best_model.pkl')) else None
        }
        
        return info