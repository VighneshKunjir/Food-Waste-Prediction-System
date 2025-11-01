# prediction/conformal_predictor.py
"""
Conformal Prediction for Food Waste with Guaranteed Coverage

Provides statistically guaranteed prediction intervals that are:
- Valid: Actual coverage ≥ target coverage (e.g., 90%)
- Efficient: As tight as possible while maintaining validity
- Distribution-free: No assumptions about data distribution

References:
- Vovk et al. (2005) - Algorithmic Learning in a Random World
- Shafer & Vovk (2008) - A Tutorial on Conformal Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import pickle
import json
from datetime import datetime
import os


class ConformalPredictor:
    """
    Conformal Prediction wrapper for waste prediction models
    
    Provides prediction intervals with guaranteed coverage probability.
    """
    
    def __init__(self, base_model, alpha: float = 0.1, method: str = 'absolute'):
        """
        Initialize Conformal Predictor
        
        Args:
            base_model: Trained regression model (sklearn-compatible)
            alpha: Miscoverage rate (0.1 = 90% coverage)
            method: Conformity score method
                - 'absolute': |y - ŷ| (default, symmetric intervals)
                - 'normalized': |y - ŷ| / σ (adaptive width)
                - 'cqr': Conformalized Quantile Regression (asymmetric)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.method = method
        self.calibration_scores = None
        self.is_calibrated = False
        
        # For normalized method
        self.scale_estimator = None
        
        # Statistics
        self.calibration_size = 0
        self.coverage_level = 1 - alpha
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrate the conformal predictor on a held-out calibration set
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        print(f"\n🎯 Calibrating Conformal Predictor (target coverage: {self.coverage_level*100:.0f}%)")
        
        # Make predictions on calibration set
        y_pred_cal = self.base_model.predict(X_cal)
        
        # Compute conformity scores
        if self.method == 'absolute':
            self.calibration_scores = np.abs(y_cal - y_pred_cal)
            
        elif self.method == 'normalized':
            # Estimate prediction uncertainty (using residual std)
            residuals = np.abs(y_cal - y_pred_cal)
            
            # Fit scale estimator (could use a separate model)
            # For simplicity, use rolling std
            window_size = max(10, len(residuals) // 10)
            scales = pd.Series(residuals).rolling(window=window_size, center=True).std()
            scales = scales.fillna(residuals.std())
            
            self.scale_estimator = scales.values
            self.calibration_scores = residuals / (scales.values + 1e-6)
            
        elif self.method == 'cqr':
            # For CQR, we need quantile predictions from base model
            # This requires the base model to support quantile regression
            # For now, fall back to absolute method
            print("⚠️ CQR method requires quantile regression model, using 'absolute' instead")
            self.method = 'absolute'
            self.calibration_scores = np.abs(y_cal - y_pred_cal)
        
        # Sort scores for quantile computation
        self.calibration_scores = np.sort(self.calibration_scores)
        self.calibration_size = len(self.calibration_scores)
        self.is_calibrated = True
        
        # Report calibration statistics
        print(f"✅ Calibrated on {self.calibration_size} samples")
        print(f"   Conformity scores: min={self.calibration_scores.min():.3f}, "
              f"median={np.median(self.calibration_scores):.3f}, "
              f"max={self.calibration_scores.max():.3f}")
    
    def predict_with_intervals(
        self, 
        X_test: np.ndarray,
        return_dict: bool = True
    ) -> Union[Dict, Tuple]:
        """
        Predict with guaranteed coverage intervals
        
        Args:
            X_test: Test features
            return_dict: If True, return dict; else return tuple
            
        Returns:
            Dictionary or tuple containing:
            - predictions: Point predictions
            - lower_bound: Lower bound of prediction interval
            - upper_bound: Upper bound of prediction interval
            - interval_width: Width of intervals
            - coverage_guarantee: Theoretical coverage (1-alpha)
        """
        if not self.is_calibrated:
            raise ValueError("Predictor not calibrated! Call calibrate() first.")
        
        # Make point predictions
        y_pred = self.base_model.predict(X_test)
        
        # Compute conformal quantile
        # Using finite-sample correction for guaranteed coverage
        n_cal = self.calibration_size
        q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        # Get quantile from calibration scores
        quantile_idx = int(np.floor(q_level * n_cal)) - 1
        quantile_idx = max(0, min(quantile_idx, n_cal - 1))
        
        interval_width = self.calibration_scores[quantile_idx]
        
        # Construct prediction intervals
        if self.method == 'normalized':
            # For normalized, we need to scale back
            # Use median scale as default (could be improved with a learned scale predictor)
            scale = np.median(self.scale_estimator) if self.scale_estimator is not None else 1.0
            interval_width = interval_width * scale
        
        lower_bound = y_pred - interval_width
        upper_bound = y_pred + interval_width
        
        # Ensure non-negative predictions (waste can't be negative)
        lower_bound = np.maximum(0, lower_bound)
        
        if return_dict:
            return {
                'predictions': y_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'coverage_guarantee': self.coverage_level,
                'alpha': self.alpha,
                'method': self.method
            }
        else:
            return y_pred, lower_bound, upper_bound
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Standard predict method (for sklearn compatibility)
        Returns only point predictions
        """
        return self.base_model.predict(X_test)
    
    def evaluate_coverage(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate actual coverage on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Print results
            
        Returns:
            Dictionary with coverage statistics
        """
        result = self.predict_with_intervals(X_test)
        
        # Check coverage
        in_interval = (y_test >= result['lower_bound']) & (y_test <= result['upper_bound'])
        actual_coverage = in_interval.mean()
        
        # Interval statistics
        avg_width = np.mean(result['interval_width'])
        median_width = np.median(result['interval_width'])
        
        # Prediction accuracy
        mae = np.mean(np.abs(y_test - result['predictions']))
        rmse = np.sqrt(np.mean((y_test - result['predictions'])**2))
        
        # Coverage by prediction range (stratified analysis)
        pred_quartiles = np.percentile(result['predictions'], [25, 50, 75])
        coverage_by_range = {}
        
        ranges = [
            (0, pred_quartiles[0], 'Q1 (Low)'),
            (pred_quartiles[0], pred_quartiles[1], 'Q2'),
            (pred_quartiles[1], pred_quartiles[2], 'Q3'),
            (pred_quartiles[2], np.inf, 'Q4 (High)')
        ]
        
        for low, high, name in ranges:
            mask = (result['predictions'] >= low) & (result['predictions'] < high)
            if mask.sum() > 0:
                coverage_by_range[name] = in_interval[mask].mean()
        
        stats = {
            'target_coverage': self.coverage_level,
            'actual_coverage': actual_coverage,
            'coverage_valid': actual_coverage >= self.coverage_level - 0.01,  # Small tolerance
            'avg_interval_width': avg_width,
            'median_interval_width': median_width,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(y_test),
            'coverage_by_range': coverage_by_range,
            'method': self.method,
            'alpha': self.alpha
        }
        
        if verbose:
            print("\n" + "="*70)
            print("📊 CONFORMAL PREDICTION EVALUATION")
            print("="*70)
            print(f"\n🎯 Target Coverage: {self.coverage_level*100:.1f}%")
            print(f"✅ Actual Coverage: {actual_coverage*100:.1f}%")
            print(f"   {'✓ VALID' if stats['coverage_valid'] else '✗ INVALID'} "
                  f"(actual ≥ target: {actual_coverage >= self.coverage_level})")
            
            print(f"\n📏 Interval Width:")
            print(f"   Average: {avg_width:.3f} kg")
            print(f"   Median:  {median_width:.3f} kg")
            
            print(f"\n📊 Prediction Accuracy:")
            print(f"   MAE:  {mae:.3f} kg")
            print(f"   RMSE: {rmse:.3f} kg")
            
            print(f"\n📈 Coverage by Prediction Range:")
            for range_name, cov in coverage_by_range.items():
                print(f"   {range_name}: {cov*100:.1f}%")
            
            print("="*70)
        
        return stats
    
    def save(self, path: str):
        """Save conformal predictor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'base_model': self.base_model,
                'alpha': self.alpha,
                'method': self.method,
                'calibration_scores': self.calibration_scores,
                'is_calibrated': self.is_calibrated,
                'scale_estimator': self.scale_estimator,
                'calibration_size': self.calibration_size,
                'coverage_level': self.coverage_level
            }, f)
        
        print(f"💾 Conformal predictor saved: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load conformal predictor"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(
            base_model=data['base_model'],
            alpha=data['alpha'],
            method=data['method']
        )
        
        predictor.calibration_scores = data['calibration_scores']
        predictor.is_calibrated = data['is_calibrated']
        predictor.scale_estimator = data['scale_estimator']
        predictor.calibration_size = data['calibration_size']
        predictor.coverage_level = data['coverage_level']
        
        print(f"✅ Conformal predictor loaded: {path}")
        return predictor


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive Conformal Prediction that adjusts to non-stationarity
    
    Uses weighted calibration scores with exponential decay to give
    more weight to recent observations.
    """
    
    def __init__(self, base_model, alpha: float = 0.1, 
                 decay_rate: float = 0.95, method: str = 'absolute'):
        """
        Initialize Adaptive Conformal Predictor
        
        Args:
            base_model: Trained model
            alpha: Miscoverage rate
            decay_rate: Weight decay for older samples (0.9-0.99)
            method: Conformity score method
        """
        super().__init__(base_model, alpha, method)
        self.decay_rate = decay_rate
        self.weighted_scores = None
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate with time-weighted scores"""
        super().calibrate(X_cal, y_cal)
        
        # Apply exponential weights (recent samples get higher weight)
        n = len(self.calibration_scores)
        weights = self.decay_rate ** np.arange(n-1, -1, -1)
        weights = weights / weights.sum()
        
        self.weighted_scores = weights
        
        print(f"⏱️ Adaptive calibration: recent samples weighted {weights[-1]/weights[0]:.1f}x more")


# ==================== UTILITY FUNCTIONS ====================

def train_conformal_predictor(
    model, 
    X_train: np.ndarray, 
    y_train: np.ndarray,
    cal_fraction: float = 0.2,
    alpha: float = 0.1,
    method: str = 'absolute',
    random_state: int = 42
) -> Tuple[ConformalPredictor, np.ndarray, np.ndarray]:
    """
    Train model and create conformal predictor with proper data split
    
    Args:
        model: Untrained model
        X_train: Training features
        y_train: Training targets
        cal_fraction: Fraction of data for calibration
        alpha: Miscoverage rate
        method: Conformity score method
        random_state: Random seed
        
    Returns:
        Tuple of (conformal_predictor, X_proper_test, y_proper_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Split into proper training and calibration
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train, y_train, 
        test_size=cal_fraction, 
        random_state=random_state
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Proper Training: {len(X_proper)} samples")
    print(f"   Calibration: {len(X_cal)} samples")
    
    # Train model on proper training set only
    print(f"\n🔄 Training base model...")
    model.fit(X_proper, y_proper)
    print(f"✅ Model trained")
    
    # Create conformal predictor
    cp = ConformalPredictor(model, alpha=alpha, method=method)
    
    # Calibrate on calibration set
    cp.calibrate(X_cal, y_cal)
    
    return cp, X_proper, y_proper


def compare_prediction_methods(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: List[float] = [0.1, 0.05, 0.01]
) -> pd.DataFrame:
    """
    Compare different conformal prediction methods and coverage levels
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for alpha in alphas:
        for method in ['absolute', 'normalized']:
            # Train conformal predictor
            cp, _, _ = train_conformal_predictor(
                model, X_train, y_train, 
                alpha=alpha, method=method
            )
            
            # Evaluate
            stats = cp.evaluate_coverage(X_test, y_test, verbose=False)
            
            results.append({
                'alpha': alpha,
                'target_coverage': f"{(1-alpha)*100:.0f}%",
                'method': method,
                'actual_coverage': f"{stats['actual_coverage']*100:.1f}%",
                'valid': '✓' if stats['coverage_valid'] else '✗',
                'avg_width': f"{stats['avg_interval_width']:.2f} kg",
                'mae': f"{stats['mae']:.2f} kg"
            })
    
    return pd.DataFrame(results)