# src/model_training_gpu.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainerGPU:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs('data/models', exist_ok=True)
        
        # Display GPU info
        if self.device.type == 'cuda':
            print(f"🔥 GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("💻 No GPU detected, using CPU")
    
    def initialize_models(self):
        """Initialize models for food waste prediction with GPU support"""
        print("🤖 Initializing Food Waste Prediction Models...")
        
        self.models = {
            # Traditional Models (CPU)
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
            # GPU-Accelerated Models
            'XGBoost (GPU)': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                tree_method='gpu_hist',  # GPU acceleration
                gpu_id=0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror',  # For regression
                eval_metric='mae'  # Mean Absolute Error for waste prediction
            ),
            
            'LightGBM (GPU)': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='regression',
                metric='mae'
            ),
        }
        
        print(f"✅ Initialized {len(self.models)} models for waste prediction")
    
    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model for waste prediction"""
        print(f"\n🔄 Training {model_name} for Food Waste Prediction...")
        
        start_time = time.time()
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Training time
            train_time = time.time() - start_time
            
            # Cross-validation for waste prediction accuracy
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            mae = -cv_scores.mean()
            std = cv_scores.std()
            
            # Calculate waste-specific metrics
            sample_predictions = model.predict(X_train[:100])
            avg_predicted_waste = np.mean(sample_predictions)
            
            print(f"   Training time: {train_time:.2f}s")
            print(f"   CV MAE: {mae:.3f} kg (+/- {std * 2:.3f})")
            print(f"   Avg predicted waste: {avg_predicted_waste:.2f} kg")
            
            # GPU memory usage for GPU models
            if 'GPU' in model_name and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU Memory used: {memory_used:.2f} GB")
            
            return model, mae, std, train_time
            
        except Exception as e:
            print(f"   ⚠️ Error training {model_name}: {e}")
            return None, float('inf'), 0, 0
    
    def train_all_models(self, X_train, y_train):
        """Train all models for food waste prediction"""
        print("\n" + "="*50)
        print("🚂 TRAINING FOOD WASTE PREDICTION MODELS")
        print("="*50)
        
        self.initialize_models()
        
        results = {}
        training_times = {'CPU': [], 'GPU': []}
        
        # Display target variable statistics
        print(f"\n📊 Waste Data Statistics:")
        print(f"   Total samples: {len(y_train)}")
        print(f"   Mean waste: {y_train.mean():.2f} kg")
        print(f"   Std waste: {y_train.std():.2f} kg")
        print(f"   Min waste: {y_train.min():.2f} kg")
        print(f"   Max waste: {y_train.max():.2f} kg")
        
        for model_name, model in self.models.items():
            trained_model, mae, std, train_time = self.train_model(
                model, X_train, y_train, model_name
            )
            
            if trained_model is not None:
                results[model_name] = {
                    'model': trained_model,
                    'mae': mae,
                    'std': std,
                    'train_time': train_time
                }
                
                # Track times
                if 'GPU' in model_name:
                    training_times['GPU'].append(train_time)
                else:
                    training_times['CPU'].append(train_time)
                
                # Track best model for waste prediction
                if mae < self.best_score:
                    self.best_score = mae
                    self.best_model = trained_model
                    self.best_model_name = model_name
        
        # Calculate speedup
        if training_times['CPU'] and training_times['GPU']:
            avg_cpu_time = np.mean(training_times['CPU'])
            avg_gpu_time = np.mean(training_times['GPU'])
            speedup = avg_cpu_time / avg_gpu_time
            print(f"\n⚡ GPU Speedup: {speedup:.2f}x faster for waste prediction")
        
        self.display_results(results)
        return results
    
    def display_results(self, results):
        """Display training results for waste prediction models"""
        print("\n" + "="*50)
        print("📊 FOOD WASTE PREDICTION MODEL COMPARISON")
        print("="*50)
        
        # Create results dataframe
        results_df = pd.DataFrame([
            {
                'Model': name,
                'MAE (kg)': data['mae'],
                'Std Dev': data['std'],
                'Training Time (s)': data['train_time'],
                'Type': 'GPU' if 'GPU' in name else 'CPU'
            }
            for name, data in results.items()
        ])
        
        results_df = results_df.sort_values('MAE (kg)')
        
        print("\n" + results_df.to_string(index=False))
        
        print(f"\n🏆 Best Waste Prediction Model: {self.best_model_name}")
        print(f"   MAE: {self.best_score:.3f} kg")
        print(f"   This means on average, predictions are off by {self.best_score:.2f} kg")
    
    def save_model(self, model, model_name):
        """Save trained waste prediction model"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/models/{model_name.replace(' ', '_')}_{timestamp}.pkl"
        
        joblib.dump(model, filename)
        print(f"💾 Waste prediction model saved: {filename}")
        
        return filename
    
    def save_all_models(self, results):
        """Save all trained models to disk"""
        import os
        import joblib
        
        os.makedirs('data/models', exist_ok=True)
        
        for model_name, model_data in results.items():
            try:
                # Create safe filename
                safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                model_path = f'data/models/{safe_name}.pkl'
                
                # Save model
                joblib.dump(model_data['model'], model_path)
                print(f"   ✅ Saved {model_name} to {model_path}")
                
            except Exception as e:
                print(f"   ⚠️ Could not save {model_name}: {e}")