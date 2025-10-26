# src/restaurant_config_manager.py
"""
Restaurant Configuration Manager
Saves and loads data format configurations for different restaurants
"""

import json
import os
from datetime import datetime

class RestaurantConfigManager:
    def __init__(self):
        self.config_dir = 'configs/restaurant_formats'
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_restaurant_config(self, restaurant_name, mapping_result, extra_info=None):
        """Save a restaurant's data format configuration"""
        
        config = {
            'restaurant_name': restaurant_name,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'column_mappings': mapping_result,
            'extra_info': extra_info or {}
        }
        
        filepath = os.path.join(self.config_dir, f'{restaurant_name}_config.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Configuration saved for {restaurant_name}")
        return filepath
    
    def load_restaurant_config(self, restaurant_name):
        """Load a saved restaurant configuration"""
        
        filepath = os.path.join(self.config_dir, f'{restaurant_name}_config.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration for {restaurant_name}")
            return config
        else:
            print(f"⚠️ No configuration found for {restaurant_name}")
            return None
    
    def list_saved_configs(self):
        """List all saved restaurant configurations"""
        
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('_config.json'):
                restaurant = file.replace('_config.json', '')
                configs.append(restaurant)
        
        return configs