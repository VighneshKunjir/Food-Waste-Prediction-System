# workflows/training_workflow.py
"""Complete training workflow orchestration"""

import os
from datetime import datetime

from core.data_loader import DataLoader
from core.data_generator import DataGenerator
from core.preprocessor import Preprocessor
from core.adapter import UniversalAdapter
from core.config_manager import ConfigManager

from training.trainer_unified import UnifiedTrainer
from training.evaluator import ModelEvaluator

from storage.model_storage import ModelStorage
from storage.restaurant_manager import RestaurantManager

from utils.gpu_utils import GPUManager
from utils.file_utils import FileManager
from utils.logger import Logger
from utils.visualization import Visualizer  # ⭐ NEW IMPORT


class TrainingWorkflow:
    """Orchestrate complete training workflow"""
    
    def __init__(self, restaurant_name, use_gpu=False, include_neural_network=True, nn_params=None):
        """
        Initialize training workflow
        
        Args:
            restaurant_name: Name of restaurant
            use_gpu: Whether to use GPU acceleration
            include_neural_network: Whether to train neural network (GPU only)
            nn_params: Dictionary of neural network parameters (optional)
        """
        self.restaurant_name = restaurant_name
        self.use_gpu = use_gpu
        self.include_neural_network = include_neural_network
        self.nn_params = nn_params  
        
        # Initialize components
        self.gpu_manager = GPUManager()
        self.logger = Logger(name=f'Training_{restaurant_name}')
        self.restaurant_manager = RestaurantManager()
        self.model_storage = ModelStorage()
        self.config_manager = ConfigManager()
        
        # Create restaurant if not exists
        if restaurant_name not in self.restaurant_manager.list_restaurants():
            self.restaurant_manager.create_restaurant(restaurant_name)
        
        # Training artifacts
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.results = None
        
        self.logger.info(f"Training workflow initialized - GPU: {use_gpu}, Include NN: {include_neural_network}")
    
    def train_with_sample_data(self, n_days=365):
        """
        Complete workflow: Generate sample data → Train → Save
        
        Args:
            n_days: Number of days of sample data to generate
        
        Returns:
            Dictionary with workflow results
        """
        print("\n" + "="*70)
        print(f"🚀 TRAINING WORKFLOW: {self.restaurant_name} (SAMPLE DATA)")
        print("="*70)
        
        self.logger.info(f"Starting training with sample data ({n_days} days)")
        
        try:
            # Step 1: Generate sample data
            print("\n1️⃣ GENERATING SAMPLE DATA")
            print("-"*70)
            
            generator = DataGenerator()
            save_path = f'data/restaurants/{self.restaurant_name}/sample_data.csv'
            FileManager.create_directory(os.path.dirname(save_path))
            
            df = generator.generate(n_days=n_days, save_path=save_path)
            self.logger.info(f"Generated {len(df)} records")
            
            # Step 2: Preprocess
            print("\n2️⃣ PREPROCESSING DATA")
            print("-"*70)
            
            self.preprocessor = Preprocessor()
            X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df)
            
            self.logger.info(f"Preprocessing complete: {X_train.shape[0]} training samples")
            
            # Step 3: Train models
            print("\n3️⃣ TRAINING MODELS")
            print("-"*70)
            
            if self.use_gpu and self.gpu_manager.gpu_available:
                self.gpu_manager.display_info()
                self.gpu_manager.optimize_gpu_settings()
            
            trainer = UnifiedTrainer(
                use_gpu=self.use_gpu,
                include_neural_network=self.include_neural_network,
                nn_params=self.nn_params)
            
            self.results = trainer.train_all_models(X_train, y_train)
            
            self.best_model = trainer.best_model
            self.best_model_name = trainer.best_model_name
            self.best_score = trainer.best_score
            
            self.logger.log_model_training(self.best_model_name, self.best_score, 0)
            
            # Step 4: Evaluate
            print("\n4️⃣ EVALUATING MODELS")
            print("-"*70)
            
            evaluator = ModelEvaluator()
            comparison = evaluator.evaluate_multiple_models(self.results, X_test, y_test)
            
            # Step 5: Save models
            print("\n5️⃣ SAVING MODELS")
            print("-"*70)
            
            self._save_training_artifacts(comparison)
            
            # ⭐ Step 6: Generate Visualizations (NEW)
            print("\n6️⃣ GENERATING VISUALIZATIONS")
            print("-"*70)
            
            self._generate_training_visualizations(df, X_train, X_test, y_train, y_test, comparison)
            
            # Step 7: Update metadata
            self._update_restaurant_metadata()
            
            # Summary
            self._display_summary()
            
            self.logger.info(f"Training workflow completed successfully")
            
            return {
                'success': True,
                'restaurant': self.restaurant_name,
                'best_model': self.best_model_name,
                'mae': self.best_score,
                'data_source': 'sample_generated',
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"\n❌ Training workflow failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def train_with_custom_csv(self, csv_path):
        """
        Complete workflow: Load CSV → Adapt → Train → Save
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Dictionary with workflow results
        """
        print("\n" + "="*70)
        print(f"🚀 TRAINING WORKFLOW: {self.restaurant_name} (CUSTOM CSV)")
        print("="*70)
        
        self.logger.info(f"Starting training with custom CSV: {csv_path}")
        
        try:
            # Step 1: Load data
            print("\n1️⃣ LOADING DATA")
            print("-"*70)
            
            loader = DataLoader()
            df = loader.load_csv(csv_path)
            loader.preview_data(df, n=3)
            
            # Step 2: Adapt data using Universal Adapter
            print("\n2️⃣ ADAPTING DATA FORMAT")
            print("-"*70)
            
            adapter = UniversalAdapter()
            
            # Check for saved config
            # Check for saved config
            saved_config = self.config_manager.load_config(self.restaurant_name)

            if saved_config:
                print(f"📋 Found saved configuration for {self.restaurant_name}")
                
                # Check if config has the required structure
                if 'column_mappings' in saved_config:
                    use_saved = input("Use saved configuration? (y/n): ").strip().lower()
                    
                    if use_saved == 'y':
                        try:
                            adapted_df = adapter._apply_mappings(df, saved_config['column_mappings'])
                            adapted_df = adapter._infer_missing(adapted_df)
                            print("✅ Applied saved configuration")
                        except Exception as e:
                            print(f"⚠️ Error applying saved config: {e}")
                            print("Re-detecting columns...")
                            adapted_df, mapping_result = adapter.detect_and_adapt(df)
                            self.config_manager.save_config(self.restaurant_name, mapping_result)
                    else:
                        # User chose not to use saved config
                        adapted_df, mapping_result = adapter.detect_and_adapt(df)
                        self.config_manager.save_config(self.restaurant_name, mapping_result)
                else:
                    # Old config format - re-detect
                    print("⚠️ Saved configuration is in old format, re-detecting columns...")
                    adapted_df, mapping_result = adapter.detect_and_adapt(df)
                    self.config_manager.save_config(self.restaurant_name, mapping_result)
            else:
                # No saved config - detect columns
                adapted_df, mapping_result = adapter.detect_and_adapt(df)
                self.config_manager.save_config(self.restaurant_name, mapping_result)
            
            # Validate adapted data
            if 'food_item' not in adapted_df.columns or 'quantity_wasted' not in adapted_df.columns:
                raise ValueError("Critical columns missing after adaptation")
            
            # Step 3: Preprocess
            print("\n3️⃣ PREPROCESSING DATA")
            print("-"*70)
            
            self.preprocessor = Preprocessor()
            X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(adapted_df)
            
            self.logger.info(f"Preprocessing complete: {X_train.shape[0]} training samples")
            
            # Step 4: Train models
            print("\n4️⃣ TRAINING MODELS")
            print("-"*70)
            
            if self.use_gpu and self.gpu_manager.gpu_available:
                self.gpu_manager.display_info()
            
            trainer = UnifiedTrainer(
                use_gpu=self.use_gpu,
                include_neural_network=self.include_neural_network,
                nn_params=self.nn_params)
            
            self.results = trainer.train_all_models(X_train, y_train)
            
            self.best_model = trainer.best_model
            self.best_model_name = trainer.best_model_name
            self.best_score = trainer.best_score
            
            # Step 5: Evaluate
            print("\n5️⃣ EVALUATING MODELS")
            print("-"*70)
            
            evaluator = ModelEvaluator()
            comparison = evaluator.evaluate_multiple_models(self.results, X_test, y_test)
            
            # Step 6: Save
            print("\n6️⃣ SAVING MODELS")
            print("-"*70)
            
            self._save_training_artifacts(comparison)
            
            # ⭐ Step 7: Generate Visualizations (NEW)
            print("\n7️⃣ GENERATING VISUALIZATIONS")
            print("-"*70)
            
            self._generate_training_visualizations(adapted_df, X_train, X_test, y_train, y_test, comparison)
            
            # Step 8: Update metadata
            self._update_restaurant_metadata()
            
            # Summary
            self._display_summary()
            
            self.logger.info(f"Training workflow completed successfully")
            
            return {
                'success': True,
                'restaurant': self.restaurant_name,
                'best_model': self.best_model_name,
                'mae': self.best_score,
                'data_source': csv_path,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.log_error_with_traceback(e)
            print(f"\n❌ Training workflow failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_training_artifacts(self, comparison_df):
        """Save all training artifacts"""
        # Prepare metadata
        metadata = {
            'model_name': self.best_model_name,
            'mae': float(self.best_score),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_best': True,
            'use_gpu': self.use_gpu,
            'neural_network_included': self.include_neural_network,
            'neural_network_params': self.nn_params,  
            'total_models_trained': len(self.results),
            'comparison': comparison_df.to_dict('records') if comparison_df is not None else None
        }
        
        # Save best model
        self.model_storage.save_model(
            model=self.best_model,
            restaurant_name=self.restaurant_name,
            metadata=metadata
        )
        
        # Save preprocessor
        self.model_storage.save_preprocessor(
            preprocessor=self.preprocessor,
            restaurant_name=self.restaurant_name
        )
        
        # Save comparison results
        if comparison_df is not None:
            results_path = f'data/restaurants/{self.restaurant_name}/training_results.csv'
            FileManager.save_csv(comparison_df, results_path)
    
    # ⭐ NEW METHOD: Generate All Training Visualizations
    def _generate_training_visualizations(self, df, X_train, X_test, y_train, y_test, comparison_df):
        """
        Generate comprehensive training visualizations
        
        Args:
            df: Original dataframe
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            comparison_df: Model comparison results
        """
        try:
            viz = Visualizer(color_theme='professional')
            
            # Create directories
            plot_dir = f'results/plots/{self.restaurant_name}'
            training_dir = f'{plot_dir}/training'
            data_dir = f'{plot_dir}/data_analysis'
            cost_dir = f'{plot_dir}/cost_analysis'
            
            os.makedirs(training_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(cost_dir, exist_ok=True)
            
            print("\n📊 Generating comprehensive visualizations...")
            
            # Get predictions for visualizations
            y_pred = self.best_model.predict(X_test)
            
            # 1. Training Dashboard (6-in-1)
            try:
                print("   1️⃣ Training Evaluation Dashboard...")
                feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else None
                viz.plot_training_dashboard(
                    model=self.best_model,
                    X_test=X_test,
                    y_test=y_test,
                    feature_names=feature_names,
                    save_path=f'{training_dir}/training_dashboard.png'
                )
            except Exception as e:
                print(f"   ⚠️ Training dashboard failed: {e}")
            
            # 2. Actual vs Predicted (individual)
            try:
                print("   2️⃣ Actual vs Predicted Plot...")
                viz.plot_actual_vs_predicted(
                    y_test, y_pred, self.best_model_name,
                    save_path=f'{training_dir}/actual_vs_predicted.png'
                )
            except Exception as e:
                print(f"   ⚠️ Actual vs Predicted failed: {e}")
            
            # 3. Residuals Analysis
            try:
                print("   3️⃣ Residuals Analysis...")
                viz.plot_residuals(
                    y_test, y_pred, self.best_model_name,
                    save_path=f'{training_dir}/residuals_analysis.png'
                )
            except Exception as e:
                print(f"   ⚠️ Residuals analysis failed: {e}")
            
            # 4. Feature Importance
            if hasattr(self.best_model, 'feature_importances_'):
                try:
                    print("   4️⃣ Feature Importance...")
                    feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else None
                    if feature_names:
                        viz.plot_feature_importance(
                            self.best_model, feature_names, top_n=15,
                            save_path=f'{training_dir}/feature_importance.png'
                        )
                except Exception as e:
                    print(f"   ⚠️ Feature importance failed: {e}")
            
            # 5. Correlation Heatmap
            try:
                print("   5️⃣ Correlation Heatmap...")
                viz.plot_correlation_heatmap(
                    df=df,
                    save_path=f'{training_dir}/correlation_heatmap.png'
                )
            except Exception as e:
                print(f"   ⚠️ Correlation heatmap failed: {e}")
            
            # 6. Model Comparison
            if comparison_df is not None and len(comparison_df) > 1:
                try:
                    print("   6️⃣ Model Comparison...")
                    viz.plot_model_comparison(
                        comparison_df,
                        save_path=f'{training_dir}/model_comparison.png'
                    )
                except Exception as e:
                    print(f"   ⚠️ Model comparison failed: {e}")
            
            # 7. Data Insights Dashboard (9-in-1)
            try:
                print("   7️⃣ Data Insights Dashboard...")
                viz.plot_data_insights_dashboard(
                    df=df,
                    save_path=f'{data_dir}/data_insights_dashboard.png'
                )
            except Exception as e:
                print(f"   ⚠️ Data insights dashboard failed: {e}")
            
            # 8-14. Individual Data Analysis Plots
            try:
                print("   8️⃣ Individual Analysis Plots...")
                
                # Waste trends
                viz.plot_waste_trends(df, ma_window=7, 
                                    save_path=f'{data_dir}/waste_trends.png')
                
                # Day of week analysis
                viz.plot_day_of_week_analysis(df, 
                                             save_path=f'{data_dir}/day_of_week_analysis.png')
                
                # Weather impact
                viz.plot_weather_impact(df, 
                                       save_path=f'{data_dir}/weather_impact.png')
                
                # Monthly patterns
                viz.plot_monthly_patterns(df, 
                                         save_path=f'{data_dir}/monthly_patterns.png')
                
                # Top wasted items
                viz.plot_top_wasted_items(df, top_n=10, 
                                         save_path=f'{data_dir}/top_wasted_items.png')
                
                # Temperature vs waste
                viz.plot_temperature_vs_waste(df, 
                                             save_path=f'{data_dir}/temperature_vs_waste.png')
                
                # Customer vs waste
                viz.plot_customer_vs_waste(df, 
                                          save_path=f'{data_dir}/customer_vs_waste.png')
                
            except Exception as e:
                print(f"   ⚠️ Some analysis plots failed: {e}")
            
            # 15-16. Cost Analysis (if cost data exists)
            if 'waste_cost' in df.columns:
                try:
                    print("   9️⃣ Cost Analysis Plots...")
                    
                    viz.plot_waste_cost_breakdown(df, 
                                                 save_path=f'{cost_dir}/cost_breakdown.png')
                    
                    viz.plot_daily_cost_trends(df, 
                                              save_path=f'{cost_dir}/daily_cost_trends.png')
                    
                except Exception as e:
                    print(f"   ⚠️ Cost analysis failed: {e}")
            
            print(f"\n✅ All visualizations generated successfully!")
            print(f"   📁 Training plots: {training_dir}")
            print(f"   📁 Data analysis: {data_dir}")
            if 'waste_cost' in df.columns:
                print(f"   📁 Cost analysis: {cost_dir}")
            
        except Exception as e:
            print(f"\n⚠️ Visualization generation encountered errors: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_restaurant_metadata(self):
        """Update restaurant metadata after training"""
        models = self.model_storage.list_models(self.restaurant_name)
        
        self.restaurant_manager.update_restaurant_metadata(
            self.restaurant_name,
            {
                'total_models': len(models),
                'last_training': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'best_model_name': self.best_model_name,
                'best_model_mae': float(self.best_score)
            }
        )
    
    def _display_summary(self):
        """Display training summary"""
        print("\n" + "="*70)
        print("✅ TRAINING WORKFLOW COMPLETE!")
        print("="*70)
        
        print(f"\n🏪 Restaurant: {self.restaurant_name}")
        print(f"🏆 Best Model: {self.best_model_name}")
        print(f"📊 MAE: {self.best_score:.3f} kg")
        print(f"💾 Models saved in: data/restaurants/{self.restaurant_name}/models/")
        print(f"📊 Visualizations saved in: results/plots/{self.restaurant_name}/")  # ⭐ NEW
        
        if self.gpu_manager.gpu_available and self.use_gpu:
            mem_info = self.gpu_manager.get_memory_info()
            if mem_info:
                print(f"🔥 GPU Memory Used: {mem_info['allocated']:.2f} GB")