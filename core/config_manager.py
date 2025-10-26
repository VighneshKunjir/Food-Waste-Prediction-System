# core/config_manager.py
"""Configuration management for restaurants"""

import json
import os
from datetime import datetime


class ConfigManager:
    """Manage restaurant configurations"""
    
    def __init__(self, config_dir='data/restaurants'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, restaurant_name, mapping_result, metadata=None):
        """Save restaurant configuration"""
        restaurant_dir = os.path.join(self.config_dir, restaurant_name)
        os.makedirs(restaurant_dir, exist_ok=True)
        
        config = {
            'restaurant_name': restaurant_name,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'column_mappings': mapping_result,
            'metadata': metadata or {}
        }
        
        config_path = os.path.join(restaurant_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Configuration saved: {config_path}")
        return config_path
    
    def load_config(self, restaurant_name):
        """Load restaurant configuration"""
        config_path = os.path.join(self.config_dir, restaurant_name, 'config.json')
        
        if not os.path.exists(config_path):
            print(f"⚠️ No configuration found for {restaurant_name}")
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded config for {restaurant_name}")
        return config
    
    def list_restaurants(self):
        """List all restaurants with saved configs"""
        if not os.path.exists(self.config_dir):
            return []
        
        restaurants = []
        for item in os.listdir(self.config_dir):
            item_path = os.path.join(self.config_dir, item)
            if os.path.isdir(item_path):
                config_file = os.path.join(item_path, 'config.json')
                if os.path.exists(config_file):
                    restaurants.append(item)
        
        return restaurants
    
    def delete_config(self, restaurant_name):
        """Delete restaurant configuration"""
        import shutil
        restaurant_dir = os.path.join(self.config_dir, restaurant_name)
        
        if os.path.exists(restaurant_dir):
            shutil.rmtree(restaurant_dir)
            print(f"✅ Deleted configuration for {restaurant_name}")
            return True
        else:
            print(f"⚠️ No configuration found for {restaurant_name}")
            return False