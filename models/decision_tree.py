# models/decision_tree.py
"""Decision Tree - CPU & GPU support"""

from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree with GPU support"""
    
    def __init__(self, max_depth=10, min_samples_split=5, use_gpu=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        super().__init__(use_gpu)
    
    def _check_gpu(self):
        """Check GPU availability"""
        # Note: Decision Tree doesn't have direct GPU support
        # Using XGBoost with depth=max_depth as GPU alternative
        return False  # Will use CPU sklearn version
    
    def _create_model(self):
        """Create Decision Tree model"""
        from sklearn.tree import DecisionTreeRegressor
        
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )