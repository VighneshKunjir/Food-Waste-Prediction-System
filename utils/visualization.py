# utils/visualization.py
"""Visualization utilities for model evaluation and predictions"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


class Visualizer:
    """Create visualizations for waste prediction system"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with plotting style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
        # Set color palette
        self.colors = sns.color_palette("husl", 10)
    
    def plot_actual_vs_predicted(self, y_true, y_pred, model_name, save_path=None):
        """Plot actual vs predicted values"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='b', linewidth=0.5)
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('Actual Waste (kg)', fontsize=12)
        ax.set_ylabel('Predicted Waste (kg)', fontsize=12)
        ax.set_title(f'{model_name}\nMAE: {mae:.2f} kg, R²: {r2:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_residuals(self, y_true, y_pred, model_name, save_path=None):
        """Plot residual analysis (4-panel)"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values (kg)')
        axes[0, 0].set_ylabel('Residuals (kg)')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Zero Error')
        axes[0, 1].set_xlabel('Residuals (kg)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals over index
        axes[1, 1].plot(residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals (kg)')
        axes[1, 1].set_title('Residuals Over Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Residuals plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=15, save_path=None):
        """Plot feature importance"""
        if not hasattr(model, 'feature_importances_'):
            print("⚠️ Model doesn't support feature importance")
            return None
        
        importances = model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_model_comparison(self, comparison_df, save_path=None):
        """Plot model comparison (4-panel)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sort by MAE
        df = comparison_df.sort_values('MAE (kg)')
        
        # 1. MAE Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(df)), df['MAE (kg)'], color='skyblue', edgecolor='navy')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax1.set_ylabel('MAE (kg)')
        ax1.set_title('Mean Absolute Error Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars1, df['MAE (kg)']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. R² Score Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(df)), df['R²'], color='lightgreen', edgecolor='darkgreen')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Comparison')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bar, val in zip(bars2, df['R²']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Accuracy Within Thresholds
        ax3 = axes[1, 0]
        x = np.arange(len(df))
        width = 0.25
        
        bars3_1 = ax3.bar(x - width, df['Within 1kg (%)'], width, 
                         label='Within 1kg', color='gold')
        bars3_2 = ax3.bar(x, df['Within 2kg (%)'], width, 
                         label='Within 2kg', color='orange')
        bars3_3 = ax3.bar(x + width, df['Within 5kg (%)'], width, 
                         label='Within 5kg', color='coral')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Prediction Accuracy Within Thresholds')
        ax3.set_xticks(x)
        ax3.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. MAPE Comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(df)), df['MAPE (%)'], color='lightcoral', edgecolor='darkred')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax4.set_ylabel('MAPE (%)')
        ax4.set_title('Mean Absolute Percentage Error')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bar, val in zip(bars4, df['MAPE (%)']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_prediction_timeline(self, dates, actual, predicted, save_path=None):
        """Plot predictions over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, actual, 'o-', label='Actual', linewidth=2, markersize=6)
        ax.plot(dates, predicted, 's--', label='Predicted', linewidth=2, markersize=6)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Waste (kg)', fontsize=12)
        ax.set_title('Waste Prediction Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Timeline plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_waste_by_category(self, df, save_path=None):
        """Plot waste distribution by category"""
        if 'food_category' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns not found")
            return None
        
        category_waste = df.groupby('food_category')['quantity_wasted'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("Set2", len(category_waste))
        bars = ax.bar(category_waste.index, category_waste.values, color=colors, edgecolor='black')
        
        ax.set_xlabel('Food Category', fontsize=12)
        ax.set_ylabel('Total Waste (kg)', fontsize=12)
        ax.set_title('Waste Distribution by Food Category', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars, category_waste.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Category plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_weekly_forecast(self, weekly_df, save_path=None):
        """Plot weekly waste forecast"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Total waste per day
        ax1.bar(weekly_df['day'], weekly_df['total_waste'], 
               color='coral', edgecolor='darkred', alpha=0.7)
        ax1.set_ylabel('Total Waste (kg)', fontsize=12)
        ax1.set_title('7-Day Waste Forecast', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for i, (day, waste) in enumerate(zip(weekly_df['day'], weekly_df['total_waste'])):
            ax1.text(i, waste + 0.5, f'{waste:.1f}', ha='center', va='bottom')
        
        # Plot 2: Cost per day
        ax2.bar(weekly_df['day'], weekly_df['total_cost'], 
               color='lightblue', edgecolor='navy', alpha=0.7)
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Waste Cost ($)', fontsize=12)
        ax2.set_title('Estimated Waste Cost', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for i, (day, cost) in enumerate(zip(weekly_df['day'], weekly_df['total_cost'])):
            ax2.text(i, cost + 0.5, f'${cost:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Weekly forecast plot saved: {save_path}")
        
        plt.show()
        
        return fig