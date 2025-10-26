# storage/__init__.py
"""Storage module - handles data persistence and management"""

from .model_storage import ModelStorage
from .prediction_storage import PredictionStorage
from .restaurant_manager import RestaurantManager

__all__ = [
    'ModelStorage',
    'PredictionStorage',
    'RestaurantManager'
]