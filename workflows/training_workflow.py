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


class TrainingWorkflow:
    """Orchestrate complete training workflow"""
    
    def __init__(self, restaurant_name, use_gpu=False, include_neural_network=True):
        """
        Initialize training workflow
        
        Args:
            restaurant_name: Name of restaurant
            use_gpu: Whether to use GPU acceleration
            include_neural_network: Whether to train neural network (GPU only)
        """
        self.restaurant_name = restaurant_name
        self.use_gpu = use_gpu
        self.include_neural_network = include_neural_network
        
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
                include_neural_network=self.include_neural_network)
            
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
            
            # Step 6: Update metadata
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
            saved_config = self.config_manager.load_config(self.restaurant_name)
            
            if saved_config:
                print(f"📋 Found saved configuration for {self.restaurant_name}")
                use_saved = input("Use saved configuration? (y/n): ").strip().lower()
                
                if use_saved == 'y':
                    adapted_df = adapter._apply_mappings(df, saved_config['column_mappings'])
                    adapted_df = adapter._infer_missing(adapted_df)
                else:
                    adapted_df, mapping_result = adapter.detect_and_adapt(df)
                    self.config_manager.save_config(self.restaurant_name, mapping_result)
            else:
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
                include_neural_network=self.include_neural_network)
            
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
        
        if self.gpu_manager.gpu_available and self.use_gpu:
            mem_info = self.gpu_manager.get_memory_info()
            if mem_info:
                print(f"🔥 GPU Memory Used: {mem_info['allocated']:.2f} GB")