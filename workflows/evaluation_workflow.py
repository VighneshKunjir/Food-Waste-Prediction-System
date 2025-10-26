# workflows/evaluation_workflow.py
"""Model evaluation workflow orchestration"""

import os
from datetime import datetime

from training.evaluator import ModelEvaluator
from storage.model_storage import ModelStorage
from storage.restaurant_manager import RestaurantManager
from utils.visualization import Visualizer
from utils.logger import Logger
from utils.file_utils import FileManager


class EvaluationWorkflow:
    """Orchestrate model evaluation workflows"""
    
    def __init__(self, restaurant_name):
        """
        Initialize evaluation workflow
        
        Args:
            restaurant_name: Name of restaurant
        """
        self.restaurant_name = restaurant_name
        
        # Initialize components
        self.logger = Logger(name=f'Evaluation_{restaurant_name}')
        self.model_storage = ModelStorage()
        self.restaurant_manager = RestaurantManager()
        self.visualizer = Visualizer()
        self.evaluator = ModelEvaluator()
        
        self.logger.info(f"Initialized evaluation workflow for {restaurant_name}")

    def _get_trained_models_info(self):
        """
        Get information about which models were trained.
        
        Returns:
            Dictionary with training metadata
        """
        try:
            metadata = self.model_storage.load_metadata(self.restaurant_name, 'best')
            if metadata:
                return {
                    'neural_network_included': metadata.get('neural_network_included', True),
                    'use_gpu': metadata.get('use_gpu', False),
                    'total_models_trained': metadata.get('total_models_trained', 0),
                    'model_name': metadata.get('model_name', 'Unknown')
                }
        except:
            pass
        
        return {
            'neural_network_included': True,
            'use_gpu': False,
            'total_models_trained': 0,
            'model_name': 'Unknown'
        }
    
    def evaluate_best_model(self, X_test, y_test, generate_plots=True):
        """
        Evaluate best model with comprehensive analysis
        
        Args:
            X_test: Test features
            y_test: Test targets
            generate_plots: Whether to generate visualizations
        
        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print(f"📊 MODEL EVALUATION: {self.restaurant_name}")
        print("="*70)
        
        try:
            # Load best model
            model = self.model_storage.load_model(self.restaurant_name, 'best')
            metadata = self.model_storage.load_metadata(self.restaurant_name, 'best')
            
            model_name = metadata.get('model_name', 'Best Model') if metadata else 'Best Model'

            # Display training info
            if metadata:
                print(f"\n📋 Model Information:")
                print(f"   Name: {model_name}")
                print(f"   GPU Used: {metadata.get('use_gpu', False)}")
                print(f"   Neural Network: {metadata.get('neural_network_included', 'Unknown')}")
                print(f"   Total Models Trained: {metadata.get('total_models_trained', 'Unknown')}")
            
            # Evaluate
            print(f"\n📈 Evaluating {model_name}...")
            metrics = self.evaluator.evaluate_model(model, X_test, y_test, model_name)
            
            # Generate plots
            if generate_plots:
                self._generate_evaluation_plots(model, X_test, y_test, model_name)
            
            # Save evaluation report
            self._save_evaluation_report(metrics)
            
            self.logger.info(f"Evaluation completed: MAE = {metrics['MAE (kg)']:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"❌ Evaluation failed: {e}")
            return None
    
    def compare_all_models(self, X_test, y_test):
        """
        Compare all saved models
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Comparison DataFrame
        """
        print("\n" + "="*70)
        print(f"📊 COMPARING ALL MODELS: {self.restaurant_name}")
        print("="*70)
        
        try:
            # List all models
            models_info = self.model_storage.list_models(self.restaurant_name)
            
            if not models_info:
                print("⚠️ No models found for comparison")
                return None
            
            print(f"\nFound {len(models_info)} models to compare")
            
            # Load and evaluate each model
            models_dict = {}
            
            for model_info in models_info[:5]:  # Limit to 5 most recent
                timestamp = model_info['timestamp']
                metadata = model_info.get('metadata', {})
                model_name = metadata.get('model_name', f'Model_{timestamp}')
                
                try:
                    model = self.model_storage.load_model(self.restaurant_name, timestamp)
                    models_dict[model_name] = model
                except Exception as e:
                    print(f"⚠️ Could not load model {model_name}: {e}")
            
            # Add best model if not already included
            try:
                best_model = self.model_storage.load_model(self.restaurant_name, 'best')
                models_dict['Best Model'] = best_model
            except:
                pass
            
            # Evaluate all models
            comparison_df = self.evaluator.evaluate_multiple_models(models_dict, X_test, y_test)
            
            # Generate comparison plot
            plot_path = f'results/plots/{self.restaurant_name}_model_comparison.png'
            FileManager.create_directory(os.path.dirname(plot_path))
            self.visualizer.plot_model_comparison(comparison_df, save_path=plot_path)
            
            # Save comparison report
            report_path = f'results/reports/{self.restaurant_name}_comparison_report.csv'
            FileManager.save_csv(comparison_df, report_path)
            
            self.logger.info(f"Model comparison completed: {len(models_dict)} models")
            
            return comparison_df
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"❌ Comparison failed: {e}")
            return None
    
    def generate_comprehensive_report(self, X_test, y_test):
        """
        Generate comprehensive evaluation report with all visualizations
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Report directory path
        """
        print("\n" + "="*70)
        print(f"📄 GENERATING COMPREHENSIVE REPORT: {self.restaurant_name}")
        print("="*70)
        
        try:
            # Create report directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = f'results/reports/{self.restaurant_name}_report_{timestamp}'
            FileManager.create_directory(report_dir)
            
            # Load best model
            model = self.model_storage.load_model(self.restaurant_name, 'best')
            metadata = self.model_storage.load_metadata(self.restaurant_name, 'best')
            model_name = metadata.get('model_name', 'Best Model') if metadata else 'Best Model'
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # 1. Actual vs Predicted
            print("\n1️⃣ Generating Actual vs Predicted plot...")
            self.visualizer.plot_actual_vs_predicted(
                y_test, y_pred, model_name,
                save_path=f'{report_dir}/actual_vs_predicted.png'
            )
            
            # 2. Residuals Analysis
            print("2️⃣ Generating Residuals analysis...")
            self.visualizer.plot_residuals(
                y_test, y_pred, model_name,
                save_path=f'{report_dir}/residuals_analysis.png'
            )
            
            # 3. Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                print("3️⃣ Generating Feature Importance...")
                from core.preprocessor import Preprocessor
                preprocessor = self.model_storage.load_preprocessor(self.restaurant_name)
                feature_names = preprocessor.feature_columns
                
                self.visualizer.plot_feature_importance(
                    model, feature_names, top_n=15,
                    save_path=f'{report_dir}/feature_importance.png'
                )
            
            # 4. Metrics Report
            print("4️⃣ Generating Metrics report...")
            metrics = self.evaluator.evaluate_model(model, X_test, y_test, model_name)
            
            import json
            with open(f'{report_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # 5. Model Info
            print("5️⃣ Saving Model information...")
            model_info = self.model_storage.get_model_info(self.restaurant_name)
            
            with open(f'{report_dir}/model_info.json', 'w') as f:
                json.dump(model_info, f, indent=4)
            
            print(f"\n✅ Comprehensive report generated: {report_dir}")
            
            self.logger.info(f"Comprehensive report generated: {report_dir}")
            
            return report_dir
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"❌ Report generation failed: {e}")
            return None
    
    def _generate_evaluation_plots(self, model, X_test, y_test, model_name):
        """Generate evaluation plots"""
        print("\n📊 Generating evaluation plots...")
        
        plot_dir = f'results/plots/{self.restaurant_name}'
        FileManager.create_directory(plot_dir)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Actual vs Predicted
        self.visualizer.plot_actual_vs_predicted(
            y_test, y_pred, model_name,
            save_path=f'{plot_dir}/actual_vs_predicted.png'
        )
        
        # Residuals
        self.visualizer.plot_residuals(
            y_test, y_pred, model_name,
            save_path=f'{plot_dir}/residuals.png'
        )
        
        print("✅ Plots generated")
    
    def _save_evaluation_report(self, metrics):
        """Save evaluation report"""
        report_path = f'results/reports/{self.restaurant_name}_evaluation.json'
        FileManager.create_directory(os.path.dirname(report_path))
        
        import json
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"💾 Evaluation report saved: {report_path}")