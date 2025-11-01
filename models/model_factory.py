# models/model_factory.py
"""
Factory to create all models - DYNAMIC VERSION
Automatically discovers and creates models from available files
"""

import importlib
import inspect
from pathlib import Path
from .base_model import BaseModel


class ModelFactory:
    """Factory to dynamically create models"""
    
    @staticmethod
    def discover_available_models():
        """
        Auto-discover available model files in the models directory.
        Returns dict with model info.
        """
        models_dir = Path(__file__).parent
        available = {}
        
        # Files to skip
        skip_files = ['__init__', 'base_model', 'model_factory', 'model_selector', 'ensemble_model']
        
        for file in sorted(models_dir.glob("*.py")):
            if file.stem in skip_files:
                continue
            
            try:
                # Try to import the module
                module = importlib.import_module(f"models.{file.stem}")
                
                # Find the model class
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseModel) and 
                        obj is not BaseModel and 
                        name.endswith('Model')):
                        
                        # Extract friendly name
                        friendly_name = name.replace('Model', '')
                        
                        # Determine if it's a neural network
                        is_nn = 'neural' in file.stem.lower() or 'nn' in file.stem.lower()
                        
                        available[friendly_name] = {
                            'class': obj,
                            'module': file.stem,
                            'class_name': name,
                            'is_neural_network': is_nn
                        }
                        
            except Exception as e:
                print(f"⚠️  Could not load {file.stem}: {e}")
        
        return available
    
    @staticmethod
    def get_all_models(use_gpu=False, include_neural_network=True, nn_params=None):
        """
        Get all available models dynamically.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            include_neural_network: Whether to include neural network models
            nn_params: Dictionary of neural network parameters (optional)
        
        Returns:
            Dictionary of model instances
        """
        available = ModelFactory.discover_available_models()
        models = {}
        
        print(f"\n📦 Creating models (GPU: {use_gpu}, Include NN: {include_neural_network})...")
        
        for friendly_name, info in available.items():
            # Skip neural network if not requested
            if info['is_neural_network'] and not include_neural_network:
                print(f"   ⏭️  Skipping {friendly_name} (Neural Network excluded)")
                continue
            
            try:
                model_class = info['class']
                
                # Create model instance with appropriate parameters
                if info['is_neural_network']:
                    # Neural network with custom or default parameters
                    if nn_params:
                        # Use custom parameters
                        print(f"   🎛️  Using custom NN parameters:")
                        print(f"      Epochs: {nn_params.get('epochs', 100)}")
                        print(f"      Batch Size: {nn_params.get('batch_size', 128)}")
                        print(f"      Learning Rate: {nn_params.get('learning_rate', 0.001)}")
                        
                        model = model_class(
                            epochs=nn_params.get('epochs', 100),
                            batch_size=nn_params.get('batch_size', 128),
                            learning_rate=nn_params.get('learning_rate', 0.001),
                            use_gpu=use_gpu
                        )
                    else:
                        # Use default parameters
                        model = model_class(
                            epochs=100,
                            batch_size=128,
                            learning_rate=0.001,
                            use_gpu=use_gpu
                        )
                elif 'Ridge' in friendly_name:
                    model = model_class(alpha=1.0, use_gpu=use_gpu)
                elif 'Lasso' in friendly_name:
                    model = model_class(alpha=0.1, use_gpu=use_gpu)
                elif 'DecisionTree' in friendly_name:
                    model = model_class(max_depth=10, use_gpu=use_gpu)
                elif 'RandomForest' in friendly_name:
                    model = model_class(n_estimators=100, max_depth=15, use_gpu=use_gpu)
                elif any(x in friendly_name for x in ['GradientBoosting', 'XGBoost', 'LightGBM']):
                    model = model_class(n_estimators=200, use_gpu=use_gpu)
                elif 'CatBoost' in friendly_name:
                    model = model_class(iterations=200, use_gpu=use_gpu)
                else:
                    # Default: just use_gpu
                    model = model_class(use_gpu=use_gpu)
                
                # Use friendly name with proper spacing
                display_name = friendly_name.replace('_', ' ')
                models[display_name] = model
                
                print(f"   ✅ Created {display_name}")
                
            except Exception as e:
                print(f"   ❌ Could not create {friendly_name}: {e}")
        
        print(f"\n✅ Total models created: {len(models)}")
        
        return models
    
    @staticmethod
    def get_model_by_name(name, use_gpu=False):
        """
        Get specific model by name.
        
        Args:
            name: Model name
            use_gpu: Whether to use GPU
        
        Returns:
            Model instance or None
        """
        models = ModelFactory.get_all_models(use_gpu=use_gpu, include_neural_network=True)
        
        # Try exact match
        if name in models:
            return models[name]
        
        # Try case-insensitive match
        for model_name, model in models.items():
            if model_name.lower() == name.lower():
                return model
        
        print(f"⚠️  Model '{name}' not found")
        return None
    
    @staticmethod
    def get_gpu_models(include_neural_network=True):
        """Get only GPU-accelerated models"""
        return ModelFactory.get_all_models(use_gpu=True, include_neural_network=include_neural_network)
    
    @staticmethod
    def get_cpu_models():
        """Get only CPU models (excludes neural network by default)"""
        return ModelFactory.get_all_models(use_gpu=False, include_neural_network=False)
    
    @staticmethod
    def list_available_models():
        """List all available models with their info"""
        available = ModelFactory.discover_available_models()
        
        model_list = []
        for friendly_name, info in available.items():
            model_list.append({
                'name': friendly_name,
                'module': info['module'],
                'class': info['class_name'],
                'is_neural_network': info['is_neural_network']
            })
        
        return model_list