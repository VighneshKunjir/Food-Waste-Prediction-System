# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
        os.makedirs('data/models', exist_ok=True)
        
    def initialize_models(self):
        """Initialize all models to be trained"""
        print("🤖 Initializing models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(alpha=1.0),
            
            'Lasso Regression': Lasso(alpha=0.1),
            
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            
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
            
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            
            'SVR': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
        }
        
        print(f"✅ Initialized {len(self.models)} models")
        
    def train_model(self, model, X_train, y_train, model_name):
        """Train a single model"""
        print(f"\n🔄 Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        mae = -cv_scores.mean()
        std = cv_scores.std()
        
        print(f"   CV MAE: {mae:.3f} (+/- {std * 2:.3f})")
        
        return model, mae, std
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best models"""
        print("\n🎯 Performing hyperparameter tuning...")
        
        # Define parameter grids for top models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            }
        }
        
        tuned_models = {}
        
        for model_name, params in param_grids.items():
            print(f"\n   Tuning {model_name}...")
            
            if model_name == 'Random Forest':
                base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif model_name == 'XGBoost':
                base_model = XGBRegressor(random_state=42, n_jobs=-1)
            
            grid_search = GridSearchCV(
                base_model,
                params,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"   Best params: {grid_search.best_params_}")
            print(f"   Best CV MAE: {-grid_search.best_score_:.3f}")
            
            tuned_models[f"{model_name} (Tuned)"] = grid_search.best_estimator_
        
        return tuned_models
    
    def train_all_models(self, X_train, y_train, perform_tuning=True):
        """Train all models and compare performance"""
        print("\n" + "="*50)
        print("🚂 TRAINING ALL MODELS")
        print("="*50)
        
        self.initialize_models()
        
        results = {}
        
        # Train base models
        for model_name, model in self.models.items():
            trained_model, mae, std = self.train_model(model, X_train, y_train, model_name)
            
            results[model_name] = {
                'model': trained_model,
                'mae': mae,
                'std': std
            }
            
            # Track best model
            if mae < self.best_score:
                self.best_score = mae
                self.best_model = trained_model
                self.best_model_name = model_name
        
        # Hyperparameter tuning
        if perform_tuning:
            tuned_models = self.hyperparameter_tuning(X_train, y_train)
            
            for model_name, model in tuned_models.items():
                trained_model, mae, std = self.train_model(model, X_train, y_train, model_name)
                
                results[model_name] = {
                    'model': trained_model,
                    'mae': mae,
                    'std': std
                }
                
                # Update best model if needed
                if mae < self.best_score:
                    self.best_score = mae
                    self.best_model = trained_model
                    self.best_model_name = model_name
        
        # Create ensemble model
        print("\n🎭 Creating Ensemble Model...")
        top_models = sorted(results.items(), key=lambda x: x[1]['mae'])[:3]
        
        from sklearn.ensemble import VotingRegressor
        ensemble = VotingRegressor([
            (name, data['model']) for name, data in top_models
        ])
        
        ensemble.fit(X_train, y_train)
        cv_scores = cross_val_score(
            ensemble, X_train, y_train, 
            cv=5, 
            scoring='neg_mean_absolute_error'
        )
        
        ensemble_mae = -cv_scores.mean()
        print(f"   Ensemble CV MAE: {ensemble_mae:.3f}")
        
        results['Ensemble'] = {
            'model': ensemble,
            'mae': ensemble_mae,
            'std': cv_scores.std()
        }
        
        if ensemble_mae < self.best_score:
            self.best_score = ensemble_mae
            self.best_model = ensemble
            self.best_model_name = 'Ensemble'
        
        self.display_results(results)
        return results
    
    def display_results(self, results):
        """Display training results in a formatted table"""
        print("\n" + "="*50)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*50)
        
        # Create results dataframe
        results_df = pd.DataFrame([
            {
                'Model': name,
                'CV MAE': data['mae'],
                'Std Dev': data['std']
            }
            for name, data in results.items()
        ])
        
        results_df = results_df.sort_values('CV MAE')
        
        print("\n" + results_df.to_string(index=False))
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   MAE: {self.best_score:.3f}")
    
    def save_model(self, model, model_name):
        """Save trained model to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/models/{model_name.replace(' ', '_')}_{timestamp}.pkl"
        
        joblib.dump(model, filename)
        print(f"💾 Model saved: {filename}")
        
        return filename
    
    def save_all_models(self, results):
        """Save all trained models"""
        print("\n💾 Saving models...")
        
        saved_models = {}
        for model_name, data in results.items():
            filename = self.save_model(data['model'], model_name)
            saved_models[model_name] = filename
        
        # Save best model separately
        best_model_path = f"data/models/best_model.pkl"
        joblib.dump(self.best_model, best_model_path)
        print(f"🏆 Best model saved: {best_model_path}")
        
        return saved_models