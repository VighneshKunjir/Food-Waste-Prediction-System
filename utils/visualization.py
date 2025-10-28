# utils/visualization.py
"""
Enhanced Visualization utilities for food waste prediction system
Includes 30+ professional visualizations and dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Create comprehensive visualizations for waste prediction system"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid', color_theme='default'):
        """
        Initialize visualizer with plotting style
        
        Args:
            style: Matplotlib style
            color_theme: 'default', 'dark', 'colorblind', 'professional'
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
        # Set color palettes based on theme
        self.color_themes = {
            'default': sns.color_palette("husl", 10),
            'dark': sns.color_palette("dark", 10),
            'colorblind': sns.color_palette("colorblind", 10),
            'professional': sns.color_palette("Set2", 10),
            'vibrant': sns.color_palette("bright", 10)
        }
        
        self.colors = self.color_themes.get(color_theme, self.color_themes['default'])
        self.primary_color = '#3498db'
        self.secondary_color = '#e74c3c'
        self.success_color = '#2ecc71'
        self.warning_color = '#f39c12'
    
    # ==================== EXISTING METHODS (KEPT AS IS) ====================
    
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
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_residuals(self, y_true, y_pred, model_name, save_path=None):
        """Plot residual analysis (4-panel)"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, color=self.primary_color)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values (kg)')
        axes[0, 0].set_ylabel('Residuals (kg)')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=self.success_color)
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
        axes[1, 1].plot(residuals, alpha=0.7, color=self.warning_color)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals (kg)')
        axes[1, 1].set_title('Residuals Over Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
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
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                      color=self.colors[:len(importance_df)])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
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
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_prediction_timeline(self, dates, actual, predicted, save_path=None):
        """Plot predictions over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, actual, 'o-', label='Actual', linewidth=2, markersize=6, 
               color=self.primary_color)
        ax.plot(dates, predicted, 's--', label='Predicted', linewidth=2, markersize=6,
               color=self.secondary_color)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Waste (kg)', fontsize=12)
        ax.set_title('Waste Prediction Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
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
            self._save_plot(save_path)
        
        plt.close()
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
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    # ==================== NEW: DATA ANALYSIS VISUALIZATIONS ====================
    
    def plot_waste_trends(self, df, ma_window=7, save_path=None):
        """
        Plot waste trends over time with moving average
        
        Args:
            df: DataFrame with 'date' and 'quantity_wasted' columns
            ma_window: Moving average window (days)
            save_path: Path to save the plot
        """
        if 'date' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns: 'date', 'quantity_wasted'")
            return None
        
        # Aggregate by date
        daily_waste = df.groupby('date')['quantity_wasted'].sum().reset_index()
        daily_waste['date'] = pd.to_datetime(daily_waste['date'])
        daily_waste = daily_waste.sort_values('date')
        
        # Calculate moving average
        daily_waste['MA'] = daily_waste['quantity_wasted'].rolling(window=ma_window).mean()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot daily waste
        ax.plot(daily_waste['date'], daily_waste['quantity_wasted'], 
               alpha=0.4, color='gray', label='Daily Waste', linewidth=1)
        
        # Plot moving average
        ax.plot(daily_waste['date'], daily_waste['MA'], 
               color=self.primary_color, linewidth=2.5, 
               label=f'{ma_window}-Day Moving Average')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Waste (kg)', fontsize=12)
        ax.set_title(f'Food Waste Trends Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_day_of_week_analysis(self, df, save_path=None):
        """
        Analyze waste patterns by day of week
        
        Args:
            df: DataFrame with 'day_of_week' and 'quantity_wasted' columns
        """
        if 'day_of_week' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns: 'day_of_week', 'quantity_wasted'")
            return None
        
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Aggregate by day
        day_waste = df.groupby('day_of_week').agg({
            'quantity_wasted': ['mean', 'std', 'sum']
        }).reset_index()
        
        day_waste.columns = ['day_of_week', 'mean_waste', 'std_waste', 'total_waste']
        
        # Sort by day order
        day_waste['day_of_week'] = pd.Categorical(day_waste['day_of_week'], 
                                                  categories=day_order, ordered=True)
        day_waste = day_waste.sort_values('day_of_week')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Average waste by day
        colors = ['#3498db' if day in ['Saturday', 'Sunday'] else '#95a5a6' 
                 for day in day_waste['day_of_week']]
        
        bars = ax1.bar(day_waste['day_of_week'], day_waste['mean_waste'], 
                      yerr=day_waste['std_waste'], capsize=5, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        ax1.set_xlabel('Day of Week', fontsize=12)
        ax1.set_ylabel('Average Waste (kg)', fontsize=12)
        ax1.set_title('Average Waste by Day of Week', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bar, val in zip(bars, day_waste['mean_waste']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Total waste by day
        bars2 = ax2.bar(day_waste['day_of_week'], day_waste['total_waste'], 
                       color=colors, edgecolor='black', alpha=0.8)
        
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Total Waste (kg)', fontsize=12)
        ax2.set_title('Total Waste by Day of Week', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Day of Week Analysis', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_weather_impact(self, df, save_path=None):
        """
        Analyze impact of weather on food waste
        
        Args:
            df: DataFrame with 'weather' and 'quantity_wasted' columns
        """
        if 'weather' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns: 'weather', 'quantity_wasted'")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Box plot
        weather_order = df.groupby('weather')['quantity_wasted'].median().sort_values(ascending=False).index
        
        sns.boxplot(data=df, y='weather', x='quantity_wasted', 
                   order=weather_order, palette='Set2', ax=ax1)
        
        ax1.set_xlabel('Waste (kg)', fontsize=12)
        ax1.set_ylabel('Weather Condition', fontsize=12)
        ax1.set_title('Waste Distribution by Weather', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Average waste by weather
        weather_stats = df.groupby('weather').agg({
            'quantity_wasted': ['mean', 'count']
        }).reset_index()
        weather_stats.columns = ['weather', 'mean_waste', 'count']
        weather_stats = weather_stats.sort_values('mean_waste', ascending=False)
        
        bars = ax2.bar(weather_stats['weather'], weather_stats['mean_waste'], 
                      color=self.colors[:len(weather_stats)], edgecolor='black', alpha=0.8)
        
        ax2.set_xlabel('Weather Condition', fontsize=12)
        ax2.set_ylabel('Average Waste (kg)', fontsize=12)
        ax2.set_title('Average Waste by Weather', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values and counts
        for bar, val, count in zip(bars, weather_stats['mean_waste'], weather_stats['count']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Weather Impact Analysis', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_monthly_patterns(self, df, save_path=None):
        """
        Analyze monthly waste patterns
        
        Args:
            df: DataFrame with 'month' or 'date' and 'quantity_wasted' columns
        """
        if 'quantity_wasted' not in df.columns:
            print("⚠️ Required column: 'quantity_wasted'")
            return None
        
        # Get month from date if month column doesn't exist
        if 'month' not in df.columns and 'date' in df.columns:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy['month'] = df_copy['date'].dt.month
        else:
            df_copy = df.copy()
        
        if 'month' not in df_copy.columns:
            print("⚠️ Required column: 'month' or 'date'")
            return None
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Aggregate by month
        monthly_waste = df_copy.groupby('month').agg({
            'quantity_wasted': ['mean', 'sum']
        }).reset_index()
        monthly_waste.columns = ['month', 'mean_waste', 'total_waste']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Line chart - average waste
        ax1.plot(monthly_waste['month'], monthly_waste['mean_waste'], 
                marker='o', linewidth=2.5, markersize=8, 
                color=self.primary_color, label='Average Waste')
        ax1.fill_between(monthly_waste['month'], monthly_waste['mean_waste'], 
                        alpha=0.3, color=self.primary_color)
        
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Average Waste (kg)', fontsize=12)
        ax1.set_title('Monthly Average Waste', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(month_names)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Bar chart - total waste
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, 12))
        bars = ax2.bar(monthly_waste['month'], monthly_waste['total_waste'], 
                      color=colors, edgecolor='black', alpha=0.8)
        
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Total Waste (kg)', fontsize=12)
        ax2.set_title('Monthly Total Waste', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(month_names)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Monthly Waste Patterns', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_top_wasted_items(self, df, top_n=10, save_path=None):
        """
        Plot top N wasted food items
        
        Args:
            df: DataFrame with 'food_item' and 'quantity_wasted' columns
            top_n: Number of top items to display
        """
        if 'food_item' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns: 'food_item', 'quantity_wasted'")
            return None
        
        # Aggregate by food item
        item_waste = df.groupby('food_item').agg({
            'quantity_wasted': ['sum', 'mean', 'count']
        }).reset_index()
        item_waste.columns = ['food_item', 'total_waste', 'avg_waste', 'count']
        
        # Get top N
        top_items = item_waste.sort_values('total_waste', ascending=False).head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Horizontal bar chart - total waste
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_items)))
        bars1 = ax1.barh(range(len(top_items)), top_items['total_waste'], 
                        color=colors, edgecolor='black')
        
        ax1.set_yticks(range(len(top_items)))
        ax1.set_yticklabels(top_items['food_item'])
        ax1.set_xlabel('Total Waste (kg)', fontsize=12)
        ax1.set_title(f'Top {top_n} Items by Total Waste', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add values
        for bar, val in zip(bars1, top_items['total_waste']):
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f} kg', va='center', fontsize=9)
        
        # Plot 2: Average waste per occurrence
        bars2 = ax2.barh(range(len(top_items)), top_items['avg_waste'], 
                        color=colors, edgecolor='black')
        
        ax2.set_yticks(range(len(top_items)))
        ax2.set_yticklabels(top_items['food_item'])
        ax2.set_xlabel('Average Waste per Day (kg)', fontsize=12)
        ax2.set_title(f'Top {top_n} Items by Average Waste', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add values
        for bar, val in zip(bars2, top_items['avg_waste']):
            ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f} kg', va='center', fontsize=9)
        
        plt.suptitle('Top Wasted Food Items Analysis', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_temperature_vs_waste(self, df, save_path=None):
        """
        Scatter plot: Temperature vs Waste
        
        Args:
            df: DataFrame with 'temperature' and 'quantity_wasted' columns
        """
        if 'temperature' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns: 'temperature', 'quantity_wasted'")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(df['temperature'], df['quantity_wasted'], 
                  alpha=0.4, s=30, color=self.primary_color, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(df['temperature'], df['quantity_wasted'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['temperature'].min(), df['temperature'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend Line (slope={z[0]:.2f})')
        
        # Calculate correlation
        corr = df['temperature'].corr(df['quantity_wasted'])
        
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Waste (kg)', fontsize=12)
        ax.set_title(f'Temperature vs Waste (Correlation: {corr:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_customer_vs_waste(self, df, save_path=None):
        """
        Scatter plot: Customer Count vs Waste
        
        Args:
            df: DataFrame with 'customer_count' and 'quantity_wasted' columns
        """
        if 'customer_count' not in df.columns or 'quantity_wasted' not in df.columns:
            print("⚠️ Required columns: 'customer_count', 'quantity_wasted'")
            return None
        
        # Aggregate by customer count
        customer_waste = df.groupby('customer_count')['quantity_wasted'].agg(['mean', 'sum', 'count']).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Scatter plot with trend
        ax1.scatter(df['customer_count'], df['quantity_wasted'], 
                   alpha=0.3, s=20, color=self.primary_color, edgecolors='black', linewidth=0.3)
        
        # Trend line
        z = np.polyfit(df['customer_count'], df['quantity_wasted'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['customer_count'].min(), df['customer_count'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=2.5, label='Trend Line')
        
        corr = df['customer_count'].corr(df['quantity_wasted'])
        
        ax1.set_xlabel('Customer Count', fontsize=12)
        ax1.set_ylabel('Waste (kg)', fontsize=12)
        ax1.set_title(f'Customers vs Waste (Correlation: {corr:.3f})', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average waste by customer bins
        customer_bins = pd.cut(df['customer_count'], bins=10)
        waste_by_bin = df.groupby(customer_bins)['quantity_wasted'].mean()
        
        bin_labels = [f'{int(interval.left)}-{int(interval.right)}' 
                     for interval in waste_by_bin.index]
        
        bars = ax2.bar(range(len(waste_by_bin)), waste_by_bin.values, 
                      color=self.success_color, edgecolor='black', alpha=0.7)
        
        ax2.set_xticks(range(len(waste_by_bin)))
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax2.set_xlabel('Customer Count Range', fontsize=12)
        ax2.set_ylabel('Average Waste (kg)', fontsize=12)
        ax2.set_title('Average Waste by Customer Range', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Customer Count vs Waste Analysis', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    # ==================== NEW: COST ANALYSIS VISUALIZATIONS ====================
    
    def plot_waste_cost_breakdown(self, df, save_path=None):
        """
        Visualize waste cost breakdown
        
        Args:
            df: DataFrame with cost-related columns
        """
        required_cols = ['food_category', 'waste_cost']
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ Required columns: {required_cols}")
            return None
        
        # Cost by category
        category_cost = df.groupby('food_category')['waste_cost'].sum().sort_values(ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Pie chart
        colors = sns.color_palette("Set3", len(category_cost))
        wedges, texts, autotexts = ax1.pie(category_cost.values, 
                                           labels=category_cost.index,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=colors,
                                           explode=[0.05] * len(category_cost))
        
        ax1.set_title('Waste Cost Distribution by Category', fontsize=13, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        # Plot 2: Bar chart with values
        bars = ax2.bar(category_cost.index, category_cost.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        ax2.set_xlabel('Food Category', fontsize=12)
        ax2.set_ylabel('Waste Cost ($)', fontsize=12)
        ax2.set_title('Waste Cost by Category', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bar, val in zip(bars, category_cost.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'${val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add total cost annotation
        total_cost = category_cost.sum()
        fig.text(0.5, 0.02, f'Total Waste Cost: ${total_cost:,.2f}', 
                ha='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Waste Cost Analysis', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_daily_cost_trends(self, df, save_path=None):
        """
        Plot daily waste cost trends
        
        Args:
            df: DataFrame with 'date' and 'waste_cost' columns
        """
        if 'date' not in df.columns or 'waste_cost' not in df.columns:
            print("⚠️ Required columns: 'date', 'waste_cost'")
            return None
        
        # Aggregate by date
        daily_cost = df.groupby('date')['waste_cost'].sum().reset_index()
        daily_cost['date'] = pd.to_datetime(daily_cost['date'])
        daily_cost = daily_cost.sort_values('date')
        
        # Calculate cumulative cost
        daily_cost['cumulative_cost'] = daily_cost['waste_cost'].cumsum()
        
        # Moving average
        daily_cost['MA_7'] = daily_cost['waste_cost'].rolling(window=7).mean()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Daily cost with moving average
        ax1.bar(daily_cost['date'], daily_cost['waste_cost'], 
               alpha=0.5, color=self.warning_color, edgecolor='darkred', label='Daily Cost')
        ax1.plot(daily_cost['date'], daily_cost['MA_7'], 
                color='darkred', linewidth=2.5, label='7-Day Moving Average')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Waste Cost ($)', fontsize=12)
        ax1.set_title('Daily Waste Cost', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Cumulative cost
        ax2.fill_between(daily_cost['date'], daily_cost['cumulative_cost'], 
                        alpha=0.3, color=self.secondary_color)
        ax2.plot(daily_cost['date'], daily_cost['cumulative_cost'], 
                color=self.secondary_color, linewidth=2.5, label='Cumulative Cost')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Cost ($)', fontsize=12)
        ax2.set_title('Cumulative Waste Cost Over Time', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add statistics
        total_cost = daily_cost['waste_cost'].sum()
        avg_daily = daily_cost['waste_cost'].mean()
        max_daily = daily_cost['waste_cost'].max()
        
        stats_text = f'Total: ${total_cost:,.2f} | Avg Daily: ${avg_daily:.2f} | Max Daily: ${max_daily:.2f}'
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Waste Cost Trends', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    # ==================== NEW: COMPREHENSIVE DASHBOARDS ====================
    
    def plot_data_insights_dashboard(self, df, save_path=None):
        """
        Comprehensive 6-panel data insights dashboard
        
        Args:
            df: Complete DataFrame with all relevant columns
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Waste by Category (Pie)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'food_category' in df.columns:
            category_waste = df.groupby('food_category')['quantity_wasted'].sum()
            ax1.pie(category_waste.values, labels=category_waste.index, autopct='%1.1f%%',
                   colors=self.colors[:len(category_waste)])
            ax1.set_title('Waste by Category', fontweight='bold')
        
        # Panel 2: Top 5 Wasted Items
        ax2 = fig.add_subplot(gs[0, 1])
        if 'food_item' in df.columns:
            top_items = df.groupby('food_item')['quantity_wasted'].sum().nlargest(5)
            ax2.barh(range(len(top_items)), top_items.values, color=self.colors[:5])
            ax2.set_yticks(range(len(top_items)))
            ax2.set_yticklabels(top_items.index)
            ax2.invert_yaxis()
            ax2.set_xlabel('Waste (kg)')
            ax2.set_title('Top 5 Wasted Items', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # Panel 3: Daily Waste Trend
        ax3 = fig.add_subplot(gs[0, 2])
        if 'date' in df.columns:
            daily = df.groupby('date')['quantity_wasted'].sum()
            ax3.plot(daily.values, color=self.primary_color, linewidth=1.5)
            ax3.fill_between(range(len(daily)), daily.values, alpha=0.3, color=self.primary_color)
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Waste (kg)')
            ax3.set_title('Daily Waste Trend', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Day of Week Pattern
        ax4 = fig.add_subplot(gs[1, 0])
        if 'day_of_week' in df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_waste = df.groupby('day_of_week')['quantity_wasted'].mean()
            
            # Reindex to correct order
            day_waste = day_waste.reindex([d for d in day_order if d in day_waste.index])
            
            colors = ['#3498db' if day in ['Saturday', 'Sunday'] else '#95a5a6' 
                     for day in day_waste.index]
            ax4.bar(range(len(day_waste)), day_waste.values, color=colors)
            ax4.set_xticks(range(len(day_waste)))
            ax4.set_xticklabels([d[:3] for d in day_waste.index])
            ax4.set_ylabel('Avg Waste (kg)')
            ax4.set_title('Average Waste by Day', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Panel 5: Weather Impact
        ax5 = fig.add_subplot(gs[1, 1])
        if 'weather' in df.columns:
            weather_waste = df.groupby('weather')['quantity_wasted'].mean().sort_values()
            ax5.barh(range(len(weather_waste)), weather_waste.values, 
                    color=self.colors[:len(weather_waste)])
            ax5.set_yticks(range(len(weather_waste)))
            ax5.set_yticklabels(weather_waste.index)
            ax5.set_xlabel('Avg Waste (kg)')
            ax5.set_title('Weather Impact', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
        
        # Panel 6: Monthly Pattern
        ax6 = fig.add_subplot(gs[1, 2])
        if 'month' in df.columns or 'date' in df.columns:
            if 'month' not in df.columns:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                df_copy['month'] = df_copy['date'].dt.month
            else:
                df_copy = df
            
            monthly = df_copy.groupby('month')['quantity_wasted'].mean()
            ax6.plot(monthly.index, monthly.values, marker='o', linewidth=2, 
                    markersize=6, color=self.success_color)
            ax6.fill_between(monthly.index, monthly.values, alpha=0.3, color=self.success_color)
            ax6.set_xlabel('Month')
            ax6.set_ylabel('Avg Waste (kg)')
            ax6.set_title('Monthly Pattern', fontweight='bold')
            ax6.set_xticks(range(1, 13))
            ax6.grid(True, alpha=0.3)
        
        # Panel 7: Temperature vs Waste
        ax7 = fig.add_subplot(gs[2, 0])
        if 'temperature' in df.columns:
            ax7.scatter(df['temperature'], df['quantity_wasted'], alpha=0.3, s=10)
            z = np.polyfit(df['temperature'], df['quantity_wasted'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['temperature'].min(), df['temperature'].max(), 100)
            ax7.plot(x_trend, p(x_trend), "r--", linewidth=2)
            ax7.set_xlabel('Temperature (°C)')
            ax7.set_ylabel('Waste (kg)')
            ax7.set_title('Temperature vs Waste', fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # Panel 8: Customer Count vs Waste
        ax8 = fig.add_subplot(gs[2, 1])
        if 'customer_count' in df.columns:
            ax8.scatter(df['customer_count'], df['quantity_wasted'], alpha=0.3, s=10)
            z = np.polyfit(df['customer_count'], df['quantity_wasted'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['customer_count'].min(), df['customer_count'].max(), 100)
            ax8.plot(x_trend, p(x_trend), "r--", linewidth=2)
            ax8.set_xlabel('Customers')
            ax8.set_ylabel('Waste (kg)')
            ax8.set_title('Customers vs Waste', fontweight='bold')
            ax8.grid(True, alpha=0.3)
        
        # Panel 9: Waste Cost Summary
        ax9 = fig.add_subplot(gs[2, 2])
        if 'waste_cost' in df.columns and 'food_category' in df.columns:
            cost_by_cat = df.groupby('food_category')['waste_cost'].sum().nlargest(5)
            ax9.bar(range(len(cost_by_cat)), cost_by_cat.values, color=self.colors[:5])
            ax9.set_xticks(range(len(cost_by_cat)))
            ax9.set_xticklabels(cost_by_cat.index, rotation=45, ha='right')
            ax9.set_ylabel('Cost ($)')
            ax9.set_title('Top Cost Categories', fontweight='bold')
            ax9.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('📊 COMPREHENSIVE DATA INSIGHTS DASHBOARD', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_training_dashboard(self, model, X_test, y_test, feature_names=None, save_path=None):
        """
        Comprehensive 6-panel training evaluation dashboard
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            save_path: Path to save
        """
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        ax1.set_xlabel('Actual (kg)')
        ax1.set_ylabel('Predicted (kg)')
        ax1.set_title(f'Actual vs Predicted\nMAE: {mae:.2f}, R²: {r2:.3f}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Residuals Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=self.success_color)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals (kg)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Feature Importance
        ax3 = fig.add_subplot(gs[0, 2])
        if hasattr(model, 'feature_importances_') and feature_names:
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10
            
            ax3.barh(range(len(indices)), importances[indices], color=self.colors)
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([feature_names[i] for i in indices])
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importance', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
        else:
            ax3.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Feature Importance', fontweight='bold')
        
        # Panel 4: Residuals vs Predicted
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(y_pred, residuals, alpha=0.5, s=30, color=self.primary_color)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted (kg)')
        ax4.set_ylabel('Residuals (kg)')
        ax4.set_title('Residuals vs Predicted', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Q-Q Plot
        ax5 = fig.add_subplot(gs[1, 1])
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot (Normality)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Error Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        abs_errors = np.abs(residuals)
        error_ranges = ['0-1kg', '1-2kg', '2-5kg', '5-10kg', '>10kg']
        counts = [
            np.sum(abs_errors <= 1),
            np.sum((abs_errors > 1) & (abs_errors <= 2)),
            np.sum((abs_errors > 2) & (abs_errors <= 5)),
            np.sum((abs_errors > 5) & (abs_errors <= 10)),
            np.sum(abs_errors > 10)
        ]
        
        colors_error = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        bars = ax6.bar(error_ranges, counts, color=colors_error, edgecolor='black', alpha=0.7)
        ax6.set_ylabel('Count')
        ax6.set_title('Error Distribution', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        total = len(residuals)
        for bar, count in zip(bars, counts):
            pct = 100 * count / total
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('🎯 TRAINING EVALUATION DASHBOARD', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_correlation_heatmap(self, df, save_path=None):
        """
        Plot correlation heatmap of numerical features
        
        Args:
            df: DataFrame
            save_path: Path to save
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            print("⚠️ Not enough numeric columns for correlation")
            return None
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    def plot_learning_curve(self, train_sizes, train_scores, val_scores, save_path=None):
        """
        Plot learning curve
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            save_path: Path to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        ax.plot(train_sizes, train_mean, 'o-', color=self.primary_color, 
               label='Training Score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color=self.primary_color)
        
        ax.plot(train_sizes, val_mean, 'o-', color=self.secondary_color, 
               label='Validation Score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color=self.secondary_color)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Score (R²)', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(save_path)
        
        plt.close()
        return fig
    
    # ==================== UTILITY METHODS ====================
    
    def _save_plot(self, save_path):
        """Save plot with proper directory creation"""
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    def set_theme(self, theme='default'):
        """Change color theme"""
        if theme in self.color_themes:
            self.colors = self.color_themes[theme]
            print(f"✅ Theme changed to: {theme}")
        else:
            print(f"⚠️ Theme '{theme}' not found. Available: {list(self.color_themes.keys())}")