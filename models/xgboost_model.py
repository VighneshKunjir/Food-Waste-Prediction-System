# models/xgboost_model.py
"""XGBoost - CPU & GPU support (Native)"""

from .base_model import BaseModel
import xgboost as xgb


class XGBoostModel(BaseModel):
    """XGBoost with native GPU support"""
    
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.1, use_gpu=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if CUDA is available for XGBoost"""
        if not self.use_gpu:
            return False
        
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _create_model(self):
        """Create XGBoost model"""
        if self.use_gpu and self.gpu_available:
            # GPU version
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                tree_method='gpu_hist',
                gpu_id=0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror',
                eval_metric='mae'
            )
        else:
            # CPU version
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                tree_method='hist',
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )