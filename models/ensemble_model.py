# models/ensemble_model.py
"""Ensemble Model - Combines top models"""

from .base_model import BaseModel
from sklearn.ensemble import VotingRegressor


class EnsembleModel(BaseModel):
    """Ensemble of top performing models"""
    
    def __init__(self, models_dict, use_gpu=False):
        """
        Args:
            models_dict: Dictionary of {name: model} to ensemble
        """
        self.models_dict = models_dict
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Ensemble uses underlying models' GPU support"""
        return any(
            getattr(model, 'gpu_available', False) 
            for model in self.models_dict.values()
        )
    
    def _create_model(self):
        """Create ensemble model"""
        estimators = [
            (name, model) 
            for name, model in self.models_dict.items()
        ]
        
        return VotingRegressor(estimators=estimators)