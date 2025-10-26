# storage/restaurant_manager.py
"""Restaurant data and configuration management"""

import os
import json
from datetime import datetime
from utils.file_utils import FileManager
from .model_storage import ModelStorage
from .prediction_storage import PredictionStorage


class RestaurantManager:
    """Manage restaurant data and operations"""
    
    def __init__(self, base_dir='data/restaurants'):
        self.base_dir = base_dir
        FileManager.create_directory(base_dir)
        
        self.model_storage = ModelStorage(base_dir)
        self.prediction_storage = PredictionStorage(base_dir)
    
    def create_restaurant(self, restaurant_name, config=None):
        """
        Create new restaurant with directory structure
        
        Args:
            restaurant_name: Name of restaurant
            config: Optional configuration dictionary
        
        Returns:
            Restaurant directory path
        """
        print(f"\n🏪 Creating restaurant: {restaurant_name}")
        
        # Create directory structure
        restaurant_dir = FileManager.get_restaurant_path(restaurant_name, create=True)
        
        # Create config
        if config is None:
            config = {}
        
        config.update({
            'restaurant_name': restaurant_name,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'active'
        })
        
        # Save config
        config_path = os.path.join(restaurant_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Create metadata
        metadata = {
            'restaurant_name': restaurant_name,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': 0,
            'total_predictions': 0,
            'last_training': None,
            'last_prediction': None
        }
        
        metadata_path = os.path.join(restaurant_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Restaurant created: {restaurant_dir}")
        
        return restaurant_dir
    
    def get_restaurant_config(self, restaurant_name):
        """Get restaurant configuration"""
        config_path = os.path.join(self.base_dir, restaurant_name, 'config.json')
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def update_restaurant_config(self, restaurant_name, config_updates):
        """Update restaurant configuration"""
        config_path = os.path.join(self.base_dir, restaurant_name, 'config.json')
        
        # Load existing config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update config
        config.update(config_updates)
        config['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Configuration updated for {restaurant_name}")
    
    def get_restaurant_metadata(self, restaurant_name):
        """Get restaurant metadata"""
        metadata_path = os.path.join(self.base_dir, restaurant_name, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def update_restaurant_metadata(self, restaurant_name, updates):
        """Update restaurant metadata"""
        metadata_path = os.path.join(self.base_dir, restaurant_name, 'metadata.json')
        
        # Load existing metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Update
        metadata.update(updates)
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def list_restaurants(self):
        """List all restaurants"""
        if not os.path.exists(self.base_dir):
            return []
        
        restaurants = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                config_path = os.path.join(item_path, 'config.json')
                if os.path.exists(config_path):
                    restaurants.append(item)
        
        return sorted(restaurants)
    
    def get_restaurant_info(self, restaurant_name):
        """Get comprehensive restaurant information"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        
        if not os.path.exists(restaurant_dir):
            return None
        
        # Get config and metadata
        config = self.get_restaurant_config(restaurant_name)
        metadata = self.get_restaurant_metadata(restaurant_name)
        
        # Get model info
        model_info = self.model_storage.get_model_info(restaurant_name)
        
        # Get prediction summary
        prediction_summary = self.prediction_storage.get_prediction_summary(restaurant_name)
        
        info = {
            'restaurant_name': restaurant_name,
            'directory': restaurant_dir,
            'config': config,
            'metadata': metadata,
            'models': model_info,
            'predictions': prediction_summary
        }
        
        return info
    
    def display_restaurant_info(self, restaurant_name):
        """Display restaurant information in formatted way"""
        info = self.get_restaurant_info(restaurant_name)
        
        if info is None:
            print(f"⚠️ Restaurant not found: {restaurant_name}")
            return
        
        print("\n" + "="*60)
        print(f"🏪 RESTAURANT INFORMATION: {restaurant_name}")
        print("="*60)
        
        # Config
        if info['config']:
            print("\n📋 Configuration:")
            print(f"   Created: {info['config'].get('created_date', 'N/A')}")
            print(f"   Status: {info['config'].get('status', 'N/A')}")
        
        # Metadata
        if info['metadata']:
            print("\n📊 Metadata:")
            print(f"   Last Updated: {info['metadata'].get('last_updated', 'N/A')}")
            print(f"   Total Models: {info['metadata'].get('total_models', 0)}")
            print(f"   Total Predictions: {info['metadata'].get('total_predictions', 0)}")
        
        # Models
        if info['models']:
            print("\n🤖 Models:")
            print(f"   Has Best Model: {'✅' if info['models']['has_best_model'] else '❌'}")
            print(f"   Has Preprocessor: {'✅' if info['models']['has_preprocessor'] else '❌'}")
            print(f"   Total Models: {info['models']['total_models']}")
            
            if info['models']['best_model_metadata']:
                best = info['models']['best_model_metadata']
                print(f"   Best Model:")
                print(f"      - Name: {best.get('model_name', 'N/A')}")
                print(f"      - MAE: {best.get('mae', 'N/A')} kg")
                print(f"      - Trained: {best.get('saved_date', 'N/A')}")
        
        # Predictions
        if info['predictions']:
            print("\n🔮 Predictions:")
            print(f"   Total: {info['predictions']['total_predictions']}")
            print(f"   Real-time: {info['predictions']['realtime_predictions']}")
            print(f"   Future: {info['predictions']['future_predictions']}")
        
        print("\n" + "="*60)
    
    def delete_restaurant(self, restaurant_name, confirm=True):
        """Delete a restaurant and all its data"""
        restaurant_dir = os.path.join(self.base_dir, restaurant_name)
        
        if not os.path.exists(restaurant_dir):
            print(f"⚠️ Restaurant not found: {restaurant_name}")
            return False
        
        if confirm:
            response = input(f"⚠️ Delete ALL data for '{restaurant_name}'? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Deletion cancelled")
                return False
        
        # Delete directory
        success = FileManager.delete_directory(restaurant_dir)
        
        if success:
            print(f"🗑️ Restaurant deleted: {restaurant_name}")
        
        return success
    
    def export_restaurant_data(self, restaurant_name, output_dir=None):
        """Export all restaurant data"""
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"results/exports/{restaurant_name}_{timestamp}"
        
        FileManager.create_directory(output_dir)
        
        print(f"\n📦 Exporting data for {restaurant_name}...")
        
        # Export config
        config = self.get_restaurant_config(restaurant_name)
        if config:
            config_path = os.path.join(output_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"   ✅ Config exported")
        
        # Export metadata
        metadata = self.get_restaurant_metadata(restaurant_name)
        if metadata:
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"   ✅ Metadata exported")
        
        # Export predictions report
        predictions_report = self.prediction_storage.export_predictions_report(
            restaurant_name,
            output_path=os.path.join(output_dir, 'predictions_report.csv')
        )
        
        print(f"\n✅ Export complete: {output_dir}")
        
        return output_dir
    
    def get_statistics(self, restaurant_name):
        """Get statistics for a restaurant"""
        # Get all predictions
        realtime_df = self.prediction_storage.get_all_predictions_df(restaurant_name, 'realtime')
        future_df = self.prediction_storage.get_all_predictions_df(restaurant_name, 'future')
        
        stats = {
            'restaurant': restaurant_name,
            'realtime_stats': {},
            'future_stats': {}
        }
        
        # Real-time statistics
        if not realtime_df.empty and 'predicted_waste' in realtime_df.columns:
            stats['realtime_stats'] = {
                'total_predictions': len(realtime_df),
                'avg_predicted_waste': realtime_df['predicted_waste'].mean(),
                'total_predicted_waste': realtime_df['predicted_waste'].sum(),
                'min_predicted_waste': realtime_df['predicted_waste'].min(),
                'max_predicted_waste': realtime_df['predicted_waste'].max()
            }
        
        # Future statistics
        if not future_df.empty and 'predicted_waste' in future_df.columns:
            stats['future_stats'] = {
                'total_predictions': len(future_df),
                'avg_predicted_waste': future_df['predicted_waste'].mean(),
                'total_predicted_waste': future_df['predicted_waste'].sum(),
                'min_predicted_waste': future_df['predicted_waste'].min(),
                'max_predicted_waste': future_df['predicted_waste'].max()
            }
        
        return stats