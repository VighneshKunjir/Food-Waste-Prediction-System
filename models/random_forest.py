# models/random_forest.py
"""Random Forest - CPU & GPU support"""

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest with GPU support via cuML"""
    
    def __init__(self, n_estimators=100, max_depth=15, use_gpu=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if cuML Random Forest is available"""
        if not self.use_gpu:
            return False
        
        try:
            import cuml
            return True
        except ImportError:
            return False
    
    def _create_model(self):
        """Create Random Forest model"""
        if self.use_gpu and self.gpu_available:
            # GPU version via cuML
            from cuml.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
        else:
            # CPU version via sklearn
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )