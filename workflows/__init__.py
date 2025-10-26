# workflows/__init__.py
"""Workflows module - orchestrates complete processes"""

from .training_workflow import TrainingWorkflow
from .prediction_workflow import PredictionWorkflow
from .evaluation_workflow import EvaluationWorkflow
from .benchmark_workflow import BenchmarkWorkflow

__all__ = [
    'TrainingWorkflow',
    'PredictionWorkflow',
    'EvaluationWorkflow',
    'BenchmarkWorkflow'
]