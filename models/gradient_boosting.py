# models/gradient_boosting.py
"""Gradient Boosting - CPU & GPU support"""

from .base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """Gradient Boosting with GPU support via CatBoost"""
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, use_gpu=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if GPU is available"""
        if not self.use_gpu:
            return False
        
        # CatBoost has native GPU support
        try:
            import catboost
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _create_model(self):
        """Create Gradient Boosting model"""
        if self.use_gpu and self.gpu_available:
            # GPU version via CatBoost
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                task_type='GPU',
                devices='0',
                verbose=False,
                random_state=42
            )
        else:
            # CPU version via sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                random_state=42
            )