# utils/file_utils.py
"""File and directory management utilities"""

import os
import shutil
import json
import joblib
from datetime import datetime
import pandas as pd


class FileManager:
    """Manage file operations"""
    
    @staticmethod
    def create_directory(path):
        """Create directory if it doesn't exist"""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"✅ Created directory: {path}")
        return path
    
    @staticmethod
    def create_project_structure(base_path='.'):
        """Create complete project directory structure"""
        directories = [
            'data/restaurants',
            'configs',
            'logs',
            'results/plots',
            'results/reports',
            'results/exports'
        ]
        
        print("\n📁 Creating project structure...")
        
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            FileManager.create_directory(full_path)
        
        print("✅ Project structure created")
    
    @staticmethod
    def get_restaurant_path(restaurant_name, create=True):
        """Get or create restaurant-specific directory"""
        base_path = f'data/restaurants/{restaurant_name}'
        
        if create:
            FileManager.create_directory(base_path)
            FileManager.create_directory(f'{base_path}/models')
            FileManager.create_directory(f'{base_path}/predictions')
            FileManager.create_directory(f'{base_path}/predictions/future')
            FileManager.create_directory(f'{base_path}/predictions/realtime')
        
        return base_path
    
    @staticmethod
    def save_model(model, filepath):
        """Save model to file"""
        try:
            # Create directory if needed
            directory = os.path.dirname(filepath)
            if directory:
                FileManager.create_directory(directory)
            
            # Save model
            joblib.dump(model, filepath)
            print(f"💾 Model saved: {filepath}")
            
            return filepath
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    @staticmethod
    def load_model(filepath):
        """Load model from file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            model = joblib.load(filepath)
            print(f"✅ Model loaded: {filepath}")
            
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    @staticmethod
    def save_json(data, filepath):
        """Save data to JSON file"""
        try:
            directory = os.path.dirname(filepath)
            if directory:
                FileManager.create_directory(directory)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"💾 JSON saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            return None
    
    @staticmethod
    def load_json(filepath):
        """Load JSON file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"JSON file not found: {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return data
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return None
    
    @staticmethod
    def save_csv(df, filepath):
        """Save DataFrame to CSV"""
        try:
            directory = os.path.dirname(filepath)
            if directory:
                FileManager.create_directory(directory)
            
            df.to_csv(filepath, index=False)
            print(f"💾 CSV saved: {filepath}")
            
            return filepath
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return None
    
    @staticmethod
    def load_csv(filepath):
        """Load CSV file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"CSV file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            print(f"✅ CSV loaded: {filepath}")
            
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None
    
    @staticmethod
    def list_files(directory, extension=None):
        """List files in directory"""
        if not os.path.exists(directory):
            return []
        
        files = []
        for file in os.listdir(directory):
            if extension:
                if file.endswith(extension):
                    files.append(file)
            else:
                files.append(file)
        
        return files
    
    @staticmethod
    def delete_file(filepath):
        """Delete a file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"🗑️ Deleted: {filepath}")
                return True
            else:
                print(f"⚠️ File not found: {filepath}")
                return False
        except Exception as e:
            print(f"❌ Error deleting file: {e}")
            return False
    
    @staticmethod
    def delete_directory(directory):
        """Delete a directory and all contents"""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"🗑️ Deleted directory: {directory}")
                return True
            else:
                print(f"⚠️ Directory not found: {directory}")
                return False
        except Exception as e:
            print(f"❌ Error deleting directory: {e}")
            return False
    
    @staticmethod
    def get_file_size(filepath):
        """Get file size in MB"""
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
    @staticmethod
    def generate_filename(prefix, extension, include_timestamp=True):
        """Generate filename with timestamp"""
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"{prefix}_{timestamp}.{extension}"
        else:
            return f"{prefix}.{extension}"
    
    @staticmethod
    def backup_file(filepath, backup_dir='backups'):
        """Create backup of a file"""
        try:
            if not os.path.exists(filepath):
                print(f"⚠️ File not found: {filepath}")
                return None
            
            # Create backup directory
            FileManager.create_directory(backup_dir)
            
            # Generate backup filename
            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy file
            shutil.copy2(filepath, backup_path)
            print(f"✅ Backup created: {backup_path}")
            
            return backup_path
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return None
    
    @staticmethod
    def clean_old_files(directory, days=30, extension=None):
        """Delete files older than specified days"""
        import time
        
        if not os.path.exists(directory):
            return 0
        
        current_time = time.time()
        deleted_count = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Skip if extension filter is set
            if extension and not filename.endswith(extension):
                continue
            
            # Check file age
            file_time = os.path.getmtime(filepath)
            age_days = (current_time - file_time) / (24 * 3600)
            
            if age_days > days:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except:
                    pass
        
        if deleted_count > 0:
            print(f"🗑️ Deleted {deleted_count} old files from {directory}")
        
        return deleted_count