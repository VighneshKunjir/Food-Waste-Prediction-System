# models/ridge_regression.py
"""Ridge Regression - CPU & GPU support"""

from .base_model import BaseModel


class RidgeRegressionModel(BaseModel):
    """Ridge Regression with GPU support via cuML"""
    
    def __init__(self, alpha=1.0, use_gpu=False):
        self.alpha = alpha
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if cuML is available"""
        if not self.use_gpu:
            return False
        
        try:
            import cuml
            return True
        except ImportError:
            return False
    
    def _create_model(self):
        """Create Ridge Regression model"""
        if self.use_gpu and self.gpu_available:
            # GPU version via cuML
            from cuml.linear_model import Ridge
            return Ridge(alpha=self.alpha)
        else:
            # CPU version via sklearn
            from sklearn.linear_model import Ridge
            return Ridge(alpha=self.alpha, random_state=42)