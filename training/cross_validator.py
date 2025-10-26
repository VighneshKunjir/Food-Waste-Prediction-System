# training/cross_validator.py
"""Cross-validation utilities"""

from sklearn.model_selection import cross_val_score, KFold
import numpy as np


class CrossValidator:
    """Handle cross-validation for models"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def validate_model(self, model, X, y, scoring='neg_mean_absolute_error'):
        """Perform cross-validation on a single model"""
        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Convert to positive if negative scoring
        if scoring.startswith('neg_'):
            scores = -scores
        
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'scores': scores
        }
    
    def validate_multiple_models(self, models_dict, X, y, scoring='neg_mean_absolute_error'):
        """Perform cross-validation on multiple models"""
        print(f"\n🔄 Cross-validating {len(models_dict)} models ({self.n_splits}-fold)...")
        
        results = {}
        
        for model_name, model_data in models_dict.items():
            # Extract model
            if isinstance(model_data, dict):
                model = model_data.get('model')
            else:
                model = model_data
            
            if model is None:
                continue
            
            print(f"   Validating {model_name}...")
            
            try:
                cv_results = self.validate_model(model, X, y, scoring)
                results[model_name] = cv_results
                
                metric_name = scoring.replace('neg_', '').upper()
                print(f"      {metric_name}: {cv_results['mean']:.3f} (+/- {cv_results['std']*2:.3f})")
                
            except Exception as e:
                print(f"      ⚠️ Failed: {e}")
        
        return results
    
    def get_best_model(self, cv_results):
        """Get best model from cross-validation results"""
        if not cv_results:
            return None, None
        
        best_model = None
        best_score = float('inf')
        
        for model_name, results in cv_results.items():
            if results['mean'] < best_score:
                best_score = results['mean']
                best_model = model_name
        
        return best_model, best_score