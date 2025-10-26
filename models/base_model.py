# models/base_model.py
"""Base class for all models"""

from abc import ABC, abstractmethod
import time


class BaseModel(ABC):
    """Base class for all waste prediction models"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.model = None
        self.model_name = self.__class__.__name__.replace('Model', '')
        self.training_time = 0
        self.gpu_available = self._check_gpu()
    
    @abstractmethod
    def _check_gpu(self):
        """Check if GPU is available for this model"""
        pass
    
    @abstractmethod
    def _create_model(self):
        """Create the model instance"""
        pass
    
    def get_model(self):
        """Get the model instance"""
        if self.model is None:
            self.model = self._create_model()
        return self.model
    
    def fit(self, X, y):
        """Train the model"""
        print(f"\n🔄 Training {self.model_name}...")
        
        if self.use_gpu:
            if self.gpu_available:
                print(f"   Using: GPU ⚡")
            else:
                print(f"   GPU not available, using CPU")
        else:
            print(f"   Using: CPU")
        
        start_time = time.time()
        
        model = self.get_model()
        model.fit(X, y)
        
        self.training_time = time.time() - start_time
        print(f"   Training time: {self.training_time:.2f}s")
        
        return model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.model.predict(X)
    
    def get_info(self):
        """Get model information"""
        return {
            'name': self.model_name,
            'gpu_enabled': self.use_gpu and self.gpu_available,
            'training_time': self.training_time
        }