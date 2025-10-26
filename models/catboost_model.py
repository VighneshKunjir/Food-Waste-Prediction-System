# models/catboost_model.py
"""CatBoost - CPU & GPU support (Native)"""

from .base_model import BaseModel
from catboost import CatBoostRegressor


class CatBoostModel(BaseModel):
    """CatBoost with native GPU support"""
    
    def __init__(self, iterations=200, depth=8, learning_rate=0.1, use_gpu=False):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if GPU is available"""
        if not self.use_gpu:
            return False
        
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _create_model(self):
        """Create CatBoost model"""
        if self.use_gpu and self.gpu_available:
            # GPU version
            return CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                task_type='GPU',
                devices='0',
                verbose=False,
                random_state=42
            )
        else:
            # CPU version
            return CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                task_type='CPU',
                thread_count=-1,
                verbose=False,
                random_state=42
            )