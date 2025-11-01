# training/trainer_unified.py
"""Unified trainer for CPU/GPU models - DYNAMIC VERSION"""

from sklearn.model_selection import cross_val_score
import numpy as np
from models.model_factory import ModelFactory
from models.model_selector import ModelSelector


class UnifiedTrainer:
    """Unified trainer for all models (CPU/GPU) with dynamic model discovery"""
    
    def __init__(self, use_gpu=False, include_neural_network=True, nn_params=None):
        """
        Initialize trainer
        
        Args:
            use_gpu: Whether to use GPU acceleration
            include_neural_network: Whether to train neural network (only relevant if use_gpu=True)
            nn_params: Dictionary of neural network parameters (optional)
        """
        self.use_gpu = use_gpu
        self.include_neural_network = include_neural_network
        self.nn_params = nn_params  # ⭐ NEW
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
    
    def train_all_models(self, X_train, y_train):
        """
        Train all available models dynamically.
        
        Args:
            X_train: Training features
            y_train: Training targets
        
        Returns:
            Dictionary of training results
        """
        print("\n" + "="*60)
        mode = "GPU" if self.use_gpu else "CPU"
        nn_status = "with NN" if self.include_neural_network else "without NN"
        print(f"🚂 TRAINING ALL MODELS ({mode} MODE, {nn_status})")
        print("="*60)
        
        # Get all available models dynamically
        models = ModelFactory.get_all_models(
        use_gpu=self.use_gpu,
        include_neural_network=self.include_neural_network,
        nn_params=self.nn_params 
        )
        
        if not models:
            print("❌ No models available to train!")
            return self.results
        
        print(f"\n📊 Training {len(models)} models...")
        
        # Train each model
        for model_name, model_wrapper in models.items():
            try:
                # Train
                trained_model = model_wrapper.fit(X_train, y_train)
                
                # Cross-validate
                cv_scores = cross_val_score(
                    trained_model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1 if not self.use_gpu else 1
                )
                
                mae = -cv_scores.mean()
                std = cv_scores.std()
                
                print(f"   CV MAE: {mae:.3f} (+/- {std*2:.3f}) kg")
                
                # Store results
                self.results[model_name] = {
                    'model': trained_model,
                    'mae': mae,
                    'std': std,
                    'training_time': model_wrapper.training_time,
                    'gpu_enabled': model_wrapper.use_gpu and model_wrapper.gpu_available
                }
                
            except Exception as e:
                print(f"   ⚠️ Failed to train {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if not self.results:
            print("\n❌ No models were successfully trained!")
            return self.results
        
        # Select best model
        self.best_model, self.best_model_name, self.best_score = ModelSelector.select_best(
            self.results,
            metric='mae'
        )
        
        # Rank models
        ModelSelector.rank_models(self.results, metric='mae')
        
        return self.results
    
    def train_specific_models(self, X_train, y_train, model_names):
        """
        Train only specific models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_names: List of model names to train
        
        Returns:
            Dictionary of training results
        """
        print(f"\n🚂 Training {len(model_names)} specific models...")
        
        for model_name in model_names:
            model_wrapper = ModelFactory.get_model_by_name(model_name, use_gpu=self.use_gpu)
            
            if model_wrapper:
                try:
                    # Train
                    trained_model = model_wrapper.fit(X_train, y_train)
                    
                    # Cross-validate
                    cv_scores = cross_val_score(
                        trained_model,
                        X_train,
                        y_train,
                        cv=5,
                        scoring='neg_mean_absolute_error'
                    )
                    
                    mae = -cv_scores.mean()
                    std = cv_scores.std()
                    
                    self.results[model_name] = {
                        'model': trained_model,
                        'mae': mae,
                        'std': std,
                        'training_time': model_wrapper.training_time
                    }
                    
                except Exception as e:
                    print(f"   ⚠️ Failed to train {model_name}: {e}")
            else:
                print(f"   ⚠️ Model {model_name} not found")
        
        return self.results