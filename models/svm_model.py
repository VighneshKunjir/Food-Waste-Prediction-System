# models/svm_model.py
"""SVM - CPU & GPU support"""

from .base_model import BaseModel


class SVMModel(BaseModel):
    """SVM with GPU support via cuML"""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, use_gpu=False):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check if cuML SVM is available"""
        if not self.use_gpu:
            return False
        
        try:
            import cuml
            return True
        except ImportError:
            return False
    
    def _create_model(self):
        """Create SVM model"""
        if self.use_gpu and self.gpu_available:
            # GPU version via cuML
            from cuml.svm import SVR
            return SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon
            )
        else:
            # CPU version via sklearn
            from sklearn.svm import SVR
            return SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon
            )