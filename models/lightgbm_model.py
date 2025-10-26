# models/lightgbm_model.py
"""LightGBM - CPU & GPU support (Native)"""

from .base_model import BaseModel
import lightgbm as lgb


class LightGBMModel(BaseModel):
    """LightGBM with native GPU support"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.1, use_gpu=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if GPU is available for LightGBM"""
        if not self.use_gpu:
            return False
        
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _create_model(self):
        """Create LightGBM model"""
        if self.use_gpu and self.gpu_available:
            # GPU version
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='regression',
                metric='mae'
            )
        else:
            # CPU version
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )