# models/lasso_regression.py
"""Lasso Regression - CPU & GPU support"""

from .base_model import BaseModel


class LassoRegressionModel(BaseModel):
    """Lasso Regression with GPU support via cuML"""
    
    def __init__(self, alpha=0.1, use_gpu=False):
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
        """Create Lasso Regression model"""
        if self.use_gpu and self.gpu_available:
            # GPU version via cuML
            from cuml.linear_model import Lasso
            return Lasso(alpha=self.alpha)
        else:
            # CPU version via sklearn
            from sklearn.linear_model import Lasso
            return Lasso(alpha=self.alpha, random_state=42)