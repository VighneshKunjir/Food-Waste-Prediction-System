# src/evaluation_gpu.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class FoodWasteEvaluator:
    def __init__(self):
        self.results = {}
        self.predictions = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model for food waste prediction"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Ensure numpy arrays
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Waste-specific metrics
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 0.001))) * 100
        
        # Accuracy within thresholds
        within_1kg = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100
        within_2kg = np.mean(np.abs(y_test - y_pred) <= 2.0) * 100
        within_5kg = np.mean(np.abs(y_test - y_pred) <= 5.0) * 100
        
        # Over/Under prediction analysis
        over_predictions = np.mean(y_pred > y_test) * 100
        under_predictions = np.mean(y_pred < y_test) * 100
        
        metrics = {
            'Model': model_name,
            'MAE (kg)': mae,
            'RMSE (kg)': rmse,
            'R²': r2,
            'MAPE (%)': mape,
            'Within 1kg (%)': within_1kg,
            'Within 2kg (%)': within_2kg,
            'Within 5kg (%)': within_5kg,
            'Over-predicted (%)': over_predictions,
            'Under-predicted (%)': under_predictions
        }
        
        # Store results
        self.results[model_name] = metrics
        self.predictions[model_name] = {'y_true': y_test, 'y_pred': y_pred}
        
        # Display metrics
        print(f"   MAE: {mae:.2f} kg (Average prediction error)")
        print(f"   RMSE: {rmse:.2f} kg")
        print(f"   R²: {r2:.3f} (Variance explained)")
        print(f"   Within 1kg accuracy: {within_1kg:.1f}%")
        print(f"   Within 2kg accuracy: {within_2kg:.1f}%")
        print(f"   Within 5kg accuracy: {within_5kg:.1f}%")
        
        return metrics
    
    def plot_model_performance(self, model_name, save_path=None):
        """Create comprehensive visualization for a single model"""
        if model_name not in self.predictions:
            print(f"No predictions found for {model_name}")
            return
        
        y_true = self.predictions[model_name]['y_true']
        y_pred = self.predictions[model_name]['y_pred']
        metrics = self.results[model_name]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Food Waste Prediction Analysis: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors='b', linewidth=0.5)
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Waste (kg)')
        ax1.set_ylabel('Predicted Waste (kg)')
        ax1.set_title('Actual vs Predicted Waste')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² text
        ax1.text(0.05, 0.95, f'R² = {metrics["R²"]:.3f}', 
                transform=ax1.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 2. Residuals Distribution
        ax2 = axes[0, 1]
        residuals = y_true - y_pred
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        ax2.set_xlabel('Prediction Error (kg)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Prediction Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error by Waste Amount
        ax3 = axes[0, 2]
        ax3.scatter(y_true, residuals, alpha=0.5, color='coral')
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Actual Waste (kg)')
        ax3.set_ylabel('Prediction Error (kg)')
        ax3.set_title('Error vs Actual Waste Amount')
        ax3.grid(True, alpha=0.3)
        
        # 4. Percentage Error
        ax4 = axes[1, 0]
        percentage_error = np.abs(residuals) / (y_true + 0.001) * 100
        ax4.hist(percentage_error[percentage_error < 100], bins=30, 
                edgecolor='black', alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Absolute Percentage Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Percentage Errors')
        ax4.grid(True, alpha=0.3)
        
        # 5. Q-Q Plot
        ax5 = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot (Normality Check)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Box
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        metrics_text = f"""
        Performance Metrics:
        
        MAE: {metrics['MAE (kg)']:.2f} kg
        RMSE: {metrics['RMSE (kg)']:.2f} kg
        R²: {metrics['R²']:.3f}
        MAPE: {metrics['MAPE (%)']:.1f}%
        
        Accuracy Thresholds:
        Within 1kg: {metrics['Within 1kg (%)']:.1f}%
        Within 2kg: {metrics['Within 2kg (%)']:.1f}%
        Within 5kg: {metrics['Within 5kg (%)']:.1f}%
        
        Prediction Bias:
        Over-predicted: {metrics['Over-predicted (%)']:.1f}%
        Under-predicted: {metrics['Under-predicted (%)']:.1f}%
        """
        
        ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Plot saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_all_models(self, save_path=None):
        """Create comparison visualization for all models"""
        if not self.results:
            print("No models to compare")
            return
        
        # Create comparison dataframe
        df = pd.DataFrame(list(self.results.values()))
        df = df.sort_values('MAE (kg)')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Food Waste Prediction Models Comparison', fontsize=16, fontweight='bold')
        
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
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 2. R² Score Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(df)), df['R²'], color='lightgreen', edgecolor='darkgreen')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Comparison')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars2, df['R²']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 3. Accuracy Within Thresholds
        ax3 = axes[1, 0]
        x = np.arange(len(df))
        width = 0.25
        
        bars3_1 = ax3.bar(x - width, df['Within 1kg (%)'], width, label='Within 1kg', color='gold')
        bars3_2 = ax3.bar(x, df['Within 2kg (%)'], width, label='Within 2kg', color='orange')
        bars3_3 = ax3.bar(x + width, df['Within 5kg (%)'], width, label='Within 5kg', color='coral')
        
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
        
        # Add values on bars
        for bar, val in zip(bars4, df['MAPE (%)']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved: {save_path}")
        
        plt.show()
        
        # Print summary table
        print("\n" + "="*60)
        print("📊 FOOD WASTE PREDICTION MODELS - FINAL RESULTS")
        print("="*60)
        print("\n" + df.to_string(index=False))
        
        # Best model summary
        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   - Predicts waste within {best_model['MAE (kg)']:.2f} kg on average")
        print(f"   - {best_model['Within 2kg (%)']:.1f}% of predictions within 2kg of actual")
        print(f"   - Explains {best_model['R²']*100:.1f}% of waste variance")
        
        return fig