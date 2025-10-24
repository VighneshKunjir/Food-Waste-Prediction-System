# src/evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handling zeros)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Additional metrics
        median_ae = np.median(np.abs(y_true - y_pred))
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Percentage of predictions within different error margins
        within_5_percent = np.mean(np.abs((y_true - y_pred) / (y_true + 0.001)) <= 0.05) * 100
        within_10_percent = np.mean(np.abs((y_true - y_pred) / (y_true + 0.001)) <= 0.10) * 100
        within_20_percent = np.mean(np.abs((y_true - y_pred) / (y_true + 0.001)) <= 0.20) * 100
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape,
            'Median AE': median_ae,
            'Max Error': max_error,
            'Within 5%': within_5_percent,
            'Within 10%': within_10_percent,
            'Within 20%': within_20_percent
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate a single model"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store predictions
        self.predictions[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, model_name)
        
        # Display metrics
        self.display_metrics(metrics)
        
        return metrics
    
    def evaluate_multiple_models(self, models, X_test, y_test):
        """Evaluate multiple models and compare"""
        print("\n" + "="*50)
        print("🔍 MODEL EVALUATION")
        print("="*50)
        
        all_metrics = []
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                model = model_data['model']
            else:
                model = model_data
                
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            all_metrics.append(metrics)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('MAE')
        
        print("\n" + "="*50)
        print("📊 MODEL COMPARISON")
        print("="*50)
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best performing model: {best_model}")
        
        return comparison_df
    
    def display_metrics(self, metrics):
        """Display metrics in a formatted way"""
        print(f"\n   Performance Metrics for {metrics['Model']}:")
        print(f"   {'─'*40}")
        print(f"   MAE:  {metrics['MAE']:.3f}")
        print(f"   RMSE: {metrics['RMSE']:.3f}")
        print(f"   R²:   {metrics['R²']:.3f}")
        print(f"   MAPE: {metrics['MAPE (%)']:.1f}%")
        print(f"   {'─'*40}")
        print(f"   Predictions within 5%:  {metrics['Within 5%']:.1f}%")
        print(f"   Predictions within 10%: {metrics['Within 10%']:.1f}%")
        print(f"   Predictions within 20%: {metrics['Within 20%']:.1f}%")
    
    def plot_predictions(self, model_name=None, save_path=None):
        """Plot actual vs predicted values"""
        
        if model_name:
            models_to_plot = [model_name]
        else:
            models_to_plot = list(self.predictions.keys())[:4]  # Top 4 models
        
        n_models = len(models_to_plot)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, model in enumerate(models_to_plot):
            if idx >= 4:
                break
                
            y_true = self.predictions[model]['y_true']
            y_pred = self.predictions[model]['y_pred']
            
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='b', linewidth=0.5)
            
            # Perfect prediction line
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Add metrics
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            ax.set_xlabel('Actual Waste')
            ax.set_ylabel('Predicted Waste')
            ax.set_title(f'{model}\nMAE: {mae:.2f}, R²: {r2:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].axis('off')
        
        plt.suptitle('Model Predictions: Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, model_name=None, save_path=None):
        """Plot residual analysis"""
        
        if model_name:
            y_true = self.predictions[model_name]['y_true']
            y_pred = self.predictions[model_name]['y_pred']
        else:
            # Use best model
            model_name = list(self.predictions.keys())[0]
            y_true = self.predictions[model_name]['y_true']
            y_pred = self.predictions[model_name]['y_pred']
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time (index)
        axes[1, 1].plot(residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Residual plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name="Model", top_n=15, save_path=None):
        """Plot feature importance for tree-based models"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances: {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Feature importance plot saved to {save_path}")
            
            plt.show()
            
            return importance_df
        else:
            print(f"⚠️ {model_name} doesn't support feature importance")
            return None