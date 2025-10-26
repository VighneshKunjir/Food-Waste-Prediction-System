# models/linear_regression.py
"""Linear Regression - CPU & GPU support"""

from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression with GPU support via cuML"""
    
    def __init__(self, use_gpu=False):
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
        """Create Linear Regression model"""
        if self.use_gpu and self.gpu_available:
            # GPU version via cuML
            from cuml.linear_model import LinearRegression
            return LinearRegression()
        else:
            # CPU version via sklearn
            from sklearn.linear_model import LinearRegression
            return LinearRegression()