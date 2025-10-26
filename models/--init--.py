# models/__init__.py
"""
GPU-ready model definitions - DYNAMIC MODEL DISCOVERY
Models are auto-discovered from files in this directory
"""

import os
import importlib
import inspect
from pathlib import Path

# Always import base model
from .base_model import BaseModel

# Dynamic model registry
AVAILABLE_MODELS = {}
MODEL_CLASSES = {}

def discover_models():
    """
    Dynamically discover and import available model files.
    This allows adding/removing models without code changes.
    """
    models_dir = Path(__file__).parent
    discovered = []
    skipped = []
    
    # Files to skip
    skip_files = ['__init__', 'base_model', 'model_factory', 'model_selector', 'ensemble_model']
    
    print("🔍 Discovering available models...")
    
    for file in sorted(models_dir.glob("*.py")):
        if file.stem in skip_files:
            continue
        
        try:
            # Dynamically import the module
            module_name = f"models.{file.stem}"
            module = importlib.import_module(module_name)
            
            # Find model classes that inherit from BaseModel
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseModel) and 
                    obj is not BaseModel and 
                    name.endswith('Model')):
                    
                    # Create friendly name
                    friendly_name = name.replace('Model', '').replace('_', ' ')
                    
                    # Register the model
                    AVAILABLE_MODELS[file.stem] = {
                        'class': obj,
                        'name': friendly_name,
                        'module': file.stem,
                        'is_neural_network': 'neural' in file.stem.lower()
                    }
                    
                    MODEL_CLASSES[name] = obj
                    discovered.append(friendly_name)
                    
        except ImportError as e:
            skipped.append(f"{file.stem} (Import Error: {e})")
        except Exception as e:
            skipped.append(f"{file.stem} (Error: {e})")
    
    print(f"✅ Discovered {len(discovered)} models: {', '.join(discovered)}")
    
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} files:")
        for skip in skipped:
            print(f"   - {skip}")
    
    return discovered

# Auto-discover models on import
_discovered = discover_models()

# Import model factory after discovery
from .model_factory import ModelFactory

# Export all
__all__ = ['BaseModel', 'ModelFactory', 'AVAILABLE_MODELS', 'MODEL_CLASSES'] + list(MODEL_CLASSES.keys())