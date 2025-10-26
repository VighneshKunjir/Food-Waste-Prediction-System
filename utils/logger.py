# utils/logger.py
"""Logging utilities for the application"""

import logging
import os
from datetime import datetime


class Logger:
    """Custom logger for food waste prediction system"""
    
    def __init__(self, name='FoodWastePrediction', log_dir='logs', level=logging.INFO):
        """Initialize logger"""
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Create logs directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler (detailed logs)
        log_filename = f"{self.log_dir}/{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler (important messages only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_model_training(self, model_name, mae, training_time):
        """Log model training results"""
        message = f"Model: {model_name} | MAE: {mae:.3f} | Time: {training_time:.2f}s"
        self.info(message)
    
    def log_prediction(self, restaurant, predicted_waste, actual_waste=None):
        """Log prediction"""
        if actual_waste is not None:
            error = abs(predicted_waste - actual_waste)
            message = f"Restaurant: {restaurant} | Predicted: {predicted_waste:.2f} | Actual: {actual_waste:.2f} | Error: {error:.2f}"
        else:
            message = f"Restaurant: {restaurant} | Predicted: {predicted_waste:.2f}"
        
        self.info(message)
    
    def log_error_with_traceback(self, error):
        """Log error with full traceback"""
        import traceback
        error_trace = traceback.format_exc()
        self.error(f"Exception occurred: {str(error)}\n{error_trace}")
    
    def create_session_log(self, session_name):
        """Create a separate log file for a specific session"""
        session_filename = f"{self.log_dir}/{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        session_handler = logging.FileHandler(session_filename)
        session_handler.setLevel(logging.DEBUG)
        session_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        self.logger.addHandler(session_handler)
        
        self.info(f"Session log created: {session_filename}")
        
        return session_filename
    
    @staticmethod
    def get_recent_logs(log_dir='logs', n=10):
        """Get recent log entries"""
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if not log_files:
            return []
        
        # Get most recent log file
        latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
        log_path = os.path.join(log_dir, latest_log)
        
        # Read last n lines
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        return lines[-n:]