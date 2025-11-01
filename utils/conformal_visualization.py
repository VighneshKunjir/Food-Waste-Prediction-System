# utils/conformal_visualization.py
"""
Visualization utilities for Conformal Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def plot_prediction_intervals(
    y_true: np.ndarray,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    n_samples: int = 100,
    save_path: Optional[str] = None
):
    """
    Plot predictions with uncertainty intervals
    
    Args:
        y_true: Actual values
        predictions: Point predictions
        lower_bound: Lower bounds
        upper_bound: Upper bounds
        n_samples: Number of samples to plot
        save_path: Path to save plot
    """
    # Convert to numpy arrays if pandas Series
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(lower_bound, 'values'):
        lower_bound = lower_bound.values
    if hasattr(upper_bound, 'values'):
        upper_bound = upper_bound.values
    
    # Sample for visualization
    n_plot = min(n_samples, len(y_true))
    idx = np.random.choice(len(y_true), n_plot, replace=False)
    idx = np.sort(idx)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(idx))
    
    # Plot intervals as shaded region
    ax.fill_between(x, lower_bound[idx], upper_bound[idx], 
                    alpha=0.3, color='skyblue', label='Prediction Interval')
    
    # Plot predictions
    ax.plot(x, predictions[idx], 'b-', linewidth=2, label='Prediction')
    
    # Plot actual values
    ax.scatter(x, y_true[idx], color='red', s=30, alpha=0.7, 
              label='Actual', zorder=5)
    
    # Highlight points outside interval
    outside = (y_true[idx] < lower_bound[idx]) | (y_true[idx] > upper_bound[idx])
    if outside.sum() > 0:
        ax.scatter(x[outside], y_true[idx][outside], 
                  color='darkred', s=50, marker='x', 
                  label=f'Outside Interval ({outside.sum()})', zorder=6)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Waste (kg)', fontsize=12)
    ax.set_title('Predictions with Conformal Intervals', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.close()
    return fig


def plot_coverage_analysis(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    predictions: np.ndarray,
    target_coverage: float = 0.9,
    save_path: Optional[str] = None
):
    """
    Analyze coverage across prediction ranges
    """
    # Convert to numpy arrays if pandas Series
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(lower_bound, 'values'):
        lower_bound = lower_bound.values
    if hasattr(upper_bound, 'values'):
        upper_bound = upper_bound.values
    
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    
    # Split into bins by prediction value
    n_bins = min(10, len(predictions) // 10)  # Adaptive number of bins
    n_bins = max(3, n_bins)  # At least 3 bins
    
    try:
        # Try to create quantile bins
        pred_bins = pd.qcut(predictions, q=n_bins, duplicates='drop')
        
        coverage_by_bin = []
        bin_labels = []
        bin_counts = []
        
        # Check if categories exist
        if hasattr(pred_bins, 'cat') and hasattr(pred_bins.cat, 'categories'):
            for bin_val in pred_bins.cat.categories:
                mask = pred_bins == bin_val
                if mask.sum() > 0:
                    coverage = in_interval[mask].mean()
                    coverage_by_bin.append(coverage)
                    bin_labels.append(f"{bin_val.left:.1f}-{bin_val.right:.1f}")
                    bin_counts.append(mask.sum())
        else:
            # Fallback to manual binning
            raise ValueError("Categories not available")
            
    except (ValueError, TypeError) as e:
        # Fallback: use simple equal-width bins
        print(f"   ⚠️ Using equal-width bins (qcut failed: {e})")
        
        min_pred = predictions.min()
        max_pred = predictions.max()
        bin_edges = np.linspace(min_pred, max_pred, n_bins + 1)
        
        coverage_by_bin = []
        bin_labels = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i+1])
            
            if mask.sum() > 0:
                coverage = in_interval[mask].mean()
                coverage_by_bin.append(coverage)
                bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}")
                bin_counts.append(mask.sum())
    
    # If still no bins, create one bin
    if len(coverage_by_bin) == 0:
        coverage_by_bin = [in_interval.mean()]
        bin_labels = [f"{predictions.min():.1f}-{predictions.max():.1f}"]
        bin_counts = [len(predictions)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Coverage by prediction range
    colors = ['green' if c >= target_coverage else 'orange' for c in coverage_by_bin]
    bars = ax1.bar(range(len(coverage_by_bin)), coverage_by_bin, color=colors, alpha=0.7)
    ax1.axhline(y=target_coverage, color='red', linestyle='--', linewidth=2, 
               label=f'Target ({target_coverage*100:.0f}%)')
    ax1.set_xlabel('Prediction Range (kg)', fontsize=11)
    ax1.set_ylabel('Actual Coverage', fontsize=11)
    ax1.set_title('Coverage by Prediction Range', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add sample counts
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        height = bar.get_height()
        y_pos = min(height + 0.02, 0.95)  # Cap at 0.95 to stay in bounds
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Interval width distribution
    interval_widths = upper_bound - lower_bound
    ax2.hist(interval_widths, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=np.median(interval_widths), color='red', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(interval_widths):.2f} kg')
    ax2.set_xlabel('Interval Width (kg)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Prediction Interval Width Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.close()
    return fig


def plot_method_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Compare different conformal prediction methods
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Coverage comparison
    methods = comparison_df['method'].unique()
    x = np.arange(len(comparison_df))
    width = 0.35
    
    for i, method in enumerate(methods):
        mask = comparison_df['method'] == method
        data = comparison_df[mask]
        offset = (i - len(methods)/2 + 0.5) * width
        
        coverage_values = [float(c.strip('%'))/100 for c in data['actual_coverage']]
        ax1.bar(x[mask] + offset, coverage_values, width, 
               label=method, alpha=0.7)
    
    target_values = [float(c.strip('%'))/100 for c in comparison_df['target_coverage'].unique()]
    for i, target in enumerate(target_values):
        ax1.axhline(y=target, color=f'C{i}', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Coverage', fontsize=11)
    ax1.set_title('Actual vs Target Coverage', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['target_coverage'].values, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Interval width comparison
    for i, method in enumerate(methods):
        mask = comparison_df['method'] == method
        data = comparison_df[mask]
        offset = (i - len(methods)/2 + 0.5) * width
        
        widths = [float(w.split()[0]) for w in data['avg_width']]
        ax2.bar(x[mask] + offset, widths, width, 
               label=method, alpha=0.7)
    
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Average Interval Width (kg)', fontsize=11)
    ax2.set_title('Interval Width Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df['target_coverage'].values, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Validity check
    valid_counts = comparison_df.groupby('method')['valid'].apply(lambda x: (x == '✓').sum())
    invalid_counts = comparison_df.groupby('method')['valid'].apply(lambda x: (x == '✗').sum())
    
    x_pos = np.arange(len(methods))
    ax3.bar(x_pos, valid_counts.values, label='Valid', color='green', alpha=0.7)
    ax3.bar(x_pos, invalid_counts.values, bottom=valid_counts.values, 
           label='Invalid', color='red', alpha=0.7)
    
    ax3.set_xlabel('Method', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Validity Check (Coverage ≥ Target)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: MAE comparison
    for i, method in enumerate(methods):
        mask = comparison_df['method'] == method
        data = comparison_df[mask]
        offset = (i - len(methods)/2 + 0.5) * width
        
        mae_values = [float(m.split()[0]) for m in data['mae']]
        ax4.bar(x[mask] + offset, mae_values, width, 
               label=method, alpha=0.7)
    
    ax4.set_xlabel('Configuration', fontsize=11)
    ax4.set_ylabel('MAE (kg)', fontsize=11)
    ax4.set_title('Prediction Accuracy (MAE)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparison_df['target_coverage'].values, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Conformal Prediction Method Comparison', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.close()
    return fig


def plot_conformal_dashboard(
    y_true: np.ndarray,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    target_coverage: float = 0.9,
    save_path: Optional[str] = None
):
    """
    Comprehensive 4-panel conformal prediction dashboard
    """
    # Convert to numpy arrays
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(lower_bound, 'values'):
        lower_bound = lower_bound.values
    if hasattr(upper_bound, 'values'):
        upper_bound = upper_bound.values
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    actual_coverage = in_interval.mean()
    interval_widths = upper_bound - lower_bound
    
    # Panel 1: Prediction intervals (sample)
    ax1 = fig.add_subplot(gs[0, 0])
    n_plot = min(100, len(y_true))
    idx = np.random.choice(len(y_true), n_plot, replace=False)
    idx = np.sort(idx)
    x = np.arange(len(idx))
    
    ax1.fill_between(x, lower_bound[idx], upper_bound[idx], 
                     alpha=0.3, color='skyblue', label='Interval')
    ax1.plot(x, predictions[idx], 'b-', linewidth=1.5, label='Prediction')
    ax1.scatter(x, y_true[idx], color='red', s=20, alpha=0.6, label='Actual')
    
    outside = (y_true[idx] < lower_bound[idx]) | (y_true[idx] > upper_bound[idx])
    if outside.sum() > 0:
        ax1.scatter(x[outside], y_true[idx][outside], 
                   color='darkred', s=40, marker='x', zorder=5)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Waste (kg)')
    ax1.set_title(f'Prediction Intervals (n={n_plot})', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Coverage statistics
    ax2 = fig.add_subplot(gs[0, 1])
    
    metrics = {
        'Target': target_coverage,
        'Actual': actual_coverage,
        'In Interval': in_interval.sum() / len(in_interval),
    }
    
    colors = ['green' if actual_coverage >= target_coverage else 'orange']
    
    bars = ax2.barh(['Coverage'], [actual_coverage], color=colors[0], alpha=0.7, height=0.3)
    ax2.axvline(x=target_coverage, color='red', linestyle='--', linewidth=2, label='Target')
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Coverage Rate')
    ax2.set_title(f'Coverage: {actual_coverage*100:.1f}% (Target: {target_coverage*100:.0f}%)', 
                 fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add text
    status = "✓ VALID" if actual_coverage >= target_coverage else "✗ INVALID"
    ax2.text(0.5, 0.5, status, transform=ax2.transAxes, 
            fontsize=20, ha='center', va='center',
            color='green' if actual_coverage >= target_coverage else 'red',
            fontweight='bold', alpha=0.3)
    
    # Panel 3: Interval width distribution
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.hist(interval_widths, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=np.median(interval_widths), color='red', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(interval_widths):.2f} kg')
    ax3.axvline(x=np.mean(interval_widths), color='orange', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(interval_widths):.2f} kg')
    
    ax3.set_xlabel('Interval Width (kg)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Interval Width Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Error analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    errors = y_true - predictions
    
    # Scatter: error vs interval width
    colors_scatter = ['green' if in_int else 'red' for in_int in in_interval]
    ax4.scatter(interval_widths, np.abs(errors), c=colors_scatter, alpha=0.5, s=20)
    
    ax4.set_xlabel('Interval Width (kg)')
    ax4.set_ylabel('Absolute Error (kg)')
    ax4.set_title('Error vs Interval Width', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='In Interval'),
        Patch(facecolor='red', alpha=0.5, label='Outside Interval')
    ]
    ax4.legend(handles=legend_elements, loc='best')
    
    plt.suptitle('Conformal Prediction Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved: {save_path}")
    
    plt.close()
    return fig