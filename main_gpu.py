# main_gpu.py - UNIFIED FOOD WASTE PREDICTION PIPELINE
"""
Comprehensive pipeline integrating all components:
- Standard & Universal data adapters
- CPU & GPU training
- Multiple evaluation modes
- Future predictions
- Real-time predictions
- Benchmarking
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import joblib

# Import all modules
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_training_gpu import ModelTrainerGPU
from src.neural_network_gpu import NeuralNetworkWrapper
from src.neural_network_waste_gpu import FoodWasteNeuralNetwork
from src.evaluation import ModelEvaluator
from src.evaluation_gpu import FoodWasteEvaluator
from src.prediction import WastePredictor
from src.future_prediction import FutureWastePredictor
from src.universal_data_adapter import UniversalDataAdapter
from src.restaurant_config_manager import RestaurantConfigManager

# Utility imports
from generate_sample_data import generate_sample_data
from benchmark_gpu import benchmark_models


class FoodWastePipeline:
    """Unified pipeline for food waste prediction system"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.restaurant_name = "default"
        
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/plots', exist_ok=True)
        os.makedirs('configs/restaurant_formats', exist_ok=True)
        
        self.display_header()
    
    def display_header(self):
        """Display system header with GPU info"""
        print("\n" + "="*70)
        print("🍔 FOOD WASTE PREDICTION SYSTEM - COMPREHENSIVE PIPELINE")
        print("="*70)
        
        if self.gpu_available:
            print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.cuda.empty_cache()
        else:
            print("\n💻 Running on CPU (GPU not detected)")
        
        print("\n" + "="*70)
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*70)
        print("📋 MAIN MENU")
        print("="*70)
        print("\n1.  🚀 Quick Start (Generate Sample Data & Train)")
        print("2.  📊 Train with Standard Format CSV")
        print("3.  🔄 Train with Any Format CSV (Universal Adapter)")
        print("4.  📈 Evaluate Existing Models")
        print("5.  🔮 Future Waste Predictions")
        print("6.  ⚡ Real-time Predictions (Interactive)")
        print("7.  🎯 GPU Performance Benchmark")
        print("8.  📋 Model Comparison & Analysis")
        print("9.  🔧 Advanced: Train Neural Network Only")
        print("10. 💾 Manage Restaurant Configurations")
        print("11. 📖 View System Documentation")
        print("0.  ❌ Exit")
        print("\n" + "="*70)
    
    # ==================== WORKFLOW 1: QUICK START ====================
    def quick_start(self):
        """Quick start with sample data"""
        print("\n" + "="*70)
        print("🚀 QUICK START - FULL PIPELINE WITH SAMPLE DATA")
        print("="*70)
        
        try:
            # Step 1: Generate sample data
            print("\n1️⃣ Generating Sample Data...")
            df = generate_sample_data(365)
            
            # Step 2: Preprocessing
            print("\n2️⃣ Preprocessing Data...")
            self.preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test, processed_df = self.preprocessor.prepare_data(
                'data/raw/food_waste_data.csv'
            )
            
            # Step 3: Model Training
            use_gpu = 'y'
            if not self.gpu_available:
                use_gpu = input("\n🔥 Train with GPU acceleration? (y/n): ").strip().lower()
            
            print("\n3️⃣ Training Models...")
            if use_gpu == 'y' and self.gpu_available:
                trainer = ModelTrainerGPU()
                evaluator = FoodWasteEvaluator()
            else:
                trainer = ModelTrainer()
                evaluator = ModelEvaluator()
            
            results = trainer.train_all_models(X_train, y_train)
            
            # Step 4: Add Neural Network
            if self.gpu_available and use_gpu == 'y':
                print("\n4️⃣ Training Neural Network...")
                nn_model = FoodWasteNeuralNetwork(epochs=100, batch_size=128)
                nn_model.fit(X_train, y_train)
                
                nn_pred = nn_model.predict(X_test)
                nn_mae = np.mean(np.abs(y_test.values - nn_pred))
                
                results['Neural Network (GPU)'] = {
                    'model': nn_model,
                    'mae': nn_mae,
                    'std': 0
                }
                print(f"   ✅ Neural Network MAE: {nn_mae:.2f} kg")
            
            # Step 5: Evaluation
            print("\n5️⃣ Evaluating Models...")
            if isinstance(evaluator, FoodWasteEvaluator):
                for model_name, model_data in results.items():
                    evaluator.evaluate_model(
                        model_data['model'], X_test, y_test, model_name
                    )
                    plot_path = f'data/plots/{model_name.replace(" ", "_")}_analysis.png'
                    evaluator.plot_model_performance(model_name, save_path=plot_path)
                
                evaluator.compare_all_models(save_path='data/plots/models_comparison.png')
            else:
                evaluator.evaluate_multiple_models(results, X_test, y_test)
                evaluator.plot_predictions(save_path='data/plots/predictions_plot.png')
                evaluator.plot_residuals(
                    model_name=trainer.best_model_name,
                    save_path='data/plots/residuals_plot.png'
                )
            
            # Step 6: Save models
            print("\n6️⃣ Saving Models...")
            self.best_model = trainer.best_model
            self.best_model_name = trainer.best_model_name
            
            joblib.dump(self.best_model, 'data/models/best_waste_model.pkl')
            joblib.dump(self.preprocessor, 'data/models/preprocessor.pkl')
            
            # Save all models
            trainer.save_all_models(results)
            
            print("\n" + "="*70)
            print("✅ QUICK START COMPLETE!")
            print("="*70)
            print(f"🏆 Best Model: {self.best_model_name}")
            print(f"📊 MAE: {trainer.best_score:.2f} kg")
            print(f"💾 Models saved in: data/models/")
            print(f"📊 Plots saved in: data/plots/")
            
        except Exception as e:
            print(f"\n❌ Error in quick start: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 2: STANDARD FORMAT ====================
    def train_standard_format(self):
        """Train with standard format CSV"""
        print("\n" + "="*70)
        print("📊 TRAIN WITH STANDARD FORMAT CSV")
        print("="*70)
        
        filepath = input("\n📁 Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        try:
            # Preview data
            df = pd.read_csv(filepath)
            print(f"\n📊 Data Preview:")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
            proceed = input("\n🚀 Proceed with training? (y/n): ").strip().lower()
            if proceed != 'y':
                return
            
            # Preprocessing
            print("\n🔧 Preprocessing...")
            self.preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test, _ = self.preprocessor.prepare_data(filepath)
            
            # Choose trainer
            use_gpu = input("\n🔥 Use GPU acceleration? (y/n): ").strip().lower()
            
            if use_gpu == 'y' and self.gpu_available:
                trainer = ModelTrainerGPU()
                evaluator = FoodWasteEvaluator()
            else:
                trainer = ModelTrainer()
                evaluator = ModelEvaluator()
            
            # Training
            print("\n🤖 Training Models...")
            results = trainer.train_all_models(X_train, y_train)
            
            # Evaluation
            print("\n📊 Evaluating...")
            if isinstance(evaluator, FoodWasteEvaluator):
                for model_name, model_data in results.items():
                    evaluator.evaluate_model(
                        model_data['model'], X_test, y_test, model_name
                    )
                evaluator.compare_all_models(save_path='data/plots/comparison.png')
            else:
                evaluator.evaluate_multiple_models(results, X_test, y_test)
            
            # Save
            self.best_model = trainer.best_model
            self.best_model_name = trainer.best_model_name
            
            joblib.dump(self.best_model, 'data/models/best_waste_model.pkl')
            joblib.dump(self.preprocessor, 'data/models/preprocessor.pkl')
            
            print(f"\n✅ Training Complete! Best Model: {self.best_model_name}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 3: UNIVERSAL ADAPTER ====================
    def train_any_format(self):
        """Train with any CSV format using universal adapter"""
        print("\n" + "="*70)
        print("🔄 UNIVERSAL CSV ADAPTER - TRAIN WITH ANY FORMAT")
        print("="*70)
        
        filepath = input("\n📁 Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        restaurant_name = input("🏪 Enter restaurant name: ").strip()
        if not restaurant_name:
            restaurant_name = "default"
        
        self.restaurant_name = restaurant_name
        
        try:
            # Initialize adapter and config manager
            adapter = UniversalDataAdapter()
            config_manager = RestaurantConfigManager()
            
            # Check for saved config
            saved_config = config_manager.load_restaurant_config(restaurant_name)
            
            if saved_config:
                use_saved = input("\n📋 Found saved config. Use it? (y/n): ").strip().lower()
                if use_saved == 'y':
                    df = pd.read_csv(filepath)
                    adapted_df = adapter.adapt_dataframe(df, saved_config['column_mappings'])
                    adapted_df = adapter.infer_missing_columns(adapted_df)
                else:
                    adapted_df, mapping_result = adapter.process_csv(filepath)
                    config_manager.save_restaurant_config(restaurant_name, mapping_result)
            else:
                adapted_df, mapping_result = adapter.process_csv(filepath)
                config_manager.save_restaurant_config(restaurant_name, mapping_result)
            
            # Display adapted data
            print(f"\n📊 Adapted Data Summary:")
            print(f"   Rows: {len(adapted_df)}")
            print(f"   Columns: {list(adapted_df.columns)}")
            print(f"\nFirst 3 rows:")
            print(adapted_df.head(3))
            
            # Validate
            if 'food_item' not in adapted_df.columns or 'quantity_wasted' not in adapted_df.columns:
                print("\n❌ Missing critical columns. Cannot proceed.")
                return
            
            proceed = input("\n🚀 Proceed with training? (y/n): ").strip().lower()
            if proceed != 'y':
                return
            
            # Save adapted data
            adapted_df.to_csv('data/raw/adapted_data.csv', index=False)
            
            # Preprocessing
            print("\n🔧 Preprocessing...")
            self.preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test, _ = self.preprocessor.prepare_data(
                'data/raw/adapted_data.csv'
            )
            
            # Choose trainer
            use_gpu = input("\n🔥 Use GPU acceleration? (y/n): ").strip().lower()
            
            if use_gpu == 'y' and self.gpu_available:
                trainer = ModelTrainerGPU()
            else:
                trainer = ModelTrainer()
            
            # Training
            print("\n🤖 Training Models...")
            results = trainer.train_all_models(X_train, y_train)
            
            # Save per-restaurant
            restaurant_dir = f'data/models/{restaurant_name}'
            os.makedirs(restaurant_dir, exist_ok=True)
            
            model_path = f'{restaurant_dir}/best_model.pkl'
            preprocessor_path = f'{restaurant_dir}/preprocessor.pkl'
            
            joblib.dump(trainer.best_model, model_path)
            joblib.dump(self.preprocessor, preprocessor_path)
            
            # Save training summary
            summary = {
                'restaurant': restaurant_name,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'best_model': trainer.best_model_name,
                'mae': float(trainer.best_score),
                'rows_trained': len(adapted_df),
                'original_file': filepath
            }
            
            with open(f'{restaurant_dir}/training_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"\n✅ Training Complete for {restaurant_name}!")
            print(f"🏆 Best Model: {trainer.best_model_name}")
            print(f"📊 MAE: {trainer.best_score:.2f} kg")
            print(f"💾 Saved to: {restaurant_dir}/")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 4: EVALUATE MODELS ====================
    def evaluate_models(self):
        """Evaluate existing models"""
        print("\n" + "="*70)
        print("📈 EVALUATE EXISTING MODELS")
        print("="*70)
        
        # List available models
        model_dirs = []
        if os.path.exists('data/models'):
            for item in os.listdir('data/models'):
                item_path = os.path.join('data/models', item)
                if os.path.isdir(item_path):
                    model_dirs.append(item)
        
        if model_dirs:
            print("\n📁 Available model sets:")
            for i, dir_name in enumerate(model_dirs, 1):
                print(f"   {i}. {dir_name}")
        
        # Check for default models
        if os.path.exists('data/models/best_waste_model.pkl'):
            print("\n   0. Default (best_waste_model.pkl)")
        
        choice = input("\n🔢 Select model set (or Enter for default): ").strip()
        
        try:
            if choice == '' or choice == '0':
                model_path = 'data/models/best_waste_model.pkl'
                preprocessor_path = 'data/models/preprocessor.pkl'
            else:
                idx = int(choice) - 1
                restaurant = model_dirs[idx]
                model_path = f'data/models/{restaurant}/best_model.pkl'
                preprocessor_path = f'data/models/{restaurant}/preprocessor.pkl'
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                return
            
            # Load model and preprocessor
            print(f"\n📂 Loading model from {model_path}...")
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            
            # Get test data
            data_path = input("\n📁 Enter test data CSV path (or Enter to skip): ").strip()
            
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                # Prepare data
                df_processed = preprocessor.prepare_prediction_data(df)
                
                if 'quantity_wasted' in df.columns:
                    y_true = df['quantity_wasted']
                    
                    # Make predictions
                    predictions = model.predict(df_processed)
                    
                    # Evaluate
                    evaluator = FoodWasteEvaluator()
                    metrics = evaluator.evaluate_model(model, df_processed, y_true, "Loaded Model")
                    evaluator.plot_model_performance("Loaded Model", save_path='data/plots/evaluation.png')
                    
                    print("\n✅ Evaluation complete! Plot saved to data/plots/evaluation.png")
                else:
                    print("\n⚠️ No 'quantity_wasted' column found. Cannot evaluate.")
                    predictions = model.predict(df_processed)
                    df['predicted_waste'] = predictions
                    output_path = data_path.replace('.csv', '_predictions.csv')
                    df.to_csv(output_path, index=False)
                    print(f"💾 Predictions saved to: {output_path}")
            else:
                print("\n✅ Model loaded successfully (no evaluation performed)")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 5: FUTURE PREDICTIONS ====================
    def future_predictions(self):
        """Future waste predictions"""
        print("\n" + "="*70)
        print("🔮 FUTURE WASTE PREDICTIONS")
        print("="*70)
        
        if not os.path.exists('data/models/best_waste_model.pkl'):
            print("\n⚠️ No trained model found!")
            print("Please train a model first (Option 1, 2, or 3)")
            return
        
        try:
            predictor = FutureWastePredictor(
                model_path='data/models/best_waste_model.pkl',
                preprocessor_path='data/models/preprocessor.pkl'
            )
            
            predictor.run_interactive_prediction()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 6: REAL-TIME PREDICTIONS ====================
    def realtime_predictions(self):
        """Real-time interactive predictions"""
        print("\n" + "="*70)
        print("⚡ REAL-TIME PREDICTIONS")
        print("="*70)
        
        if not os.path.exists('data/models/best_waste_model.pkl'):
            print("\n⚠️ No trained model found!")
            return
        
        try:
            predictor = WastePredictor('data/models/best_waste_model.pkl')
            preprocessor = joblib.load('data/models/preprocessor.pkl')
            predictor.load_preprocessor(preprocessor)
            
            predictor.real_time_prediction()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 7: BENCHMARK ====================
    def run_benchmark(self):
        """Run GPU benchmark"""
        print("\n" + "="*70)
        print("🎯 GPU PERFORMANCE BENCHMARK")
        print("="*70)
        
        if not self.gpu_available:
            print("\n⚠️ GPU not available. Benchmark will compare CPU models only.")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return
        
        try:
            results = benchmark_models()
            print("\n✅ Benchmark complete! Results saved as 'gpu_benchmark_results.png'")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 8: MODEL COMPARISON ====================
    def model_comparison(self):
        """Compare multiple saved models"""
        print("\n" + "="*70)
        print("📋 MODEL COMPARISON & ANALYSIS")
        print("="*70)
        
        # Find all saved models
        model_files = []
        for root, dirs, files in os.walk('data/models'):
            for file in files:
                if file.endswith('.pkl') and 'model' in file.lower():
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            print("\n⚠️ No saved models found!")
            return
        
        print(f"\n📁 Found {len(model_files)} model files:")
        for i, path in enumerate(model_files, 1):
            print(f"   {i}. {path}")
        
        # Load test data
        data_path = input("\n📁 Enter test data CSV path: ").strip().strip('"')
        
        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            return
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            if 'quantity_wasted' not in df.columns:
                print("❌ Test data must contain 'quantity_wasted' column")
                return
            
            # Load preprocessor
            preprocessor_path = 'data/models/preprocessor.pkl'
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
                X_test = preprocessor.prepare_prediction_data(df)
                y_test = df['quantity_wasted']
            else:
                print("❌ Preprocessor not found")
                return
            
            # Compare models
            evaluator = FoodWasteEvaluator()
            
            for i, model_path in enumerate(model_files[:5], 1):  # Limit to 5
                try:
                    model = joblib.load(model_path)
                    model_name = os.path.basename(model_path).replace('.pkl', '')
                    
                    evaluator.evaluate_model(model, X_test, y_test, model_name)
                    
                except Exception as e:
                    print(f"⚠️ Could not evaluate {model_path}: {e}")
            
            # Generate comparison plot
            evaluator.compare_all_models(save_path='data/plots/model_comparison.png')
            
            print("\n✅ Comparison complete! Plot saved to data/plots/model_comparison.png")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 9: NEURAL NETWORK ONLY ====================
    def train_neural_network_only(self):
        """Train only neural network"""
        print("\n" + "="*70)
        print("🔧 ADVANCED: TRAIN NEURAL NETWORK ONLY")
        print("="*70)
        
        if not self.gpu_available:
            print("\n⚠️ GPU not available. Neural network training will be slow on CPU.")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return
        
        data_path = input("\n📁 Enter training data CSV path: ").strip().strip('"')
        
        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            return
        
        try:
            # Preprocessing
            print("\n🔧 Preprocessing...")
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data(data_path)
            
            # Get hyperparameters
            epochs = input("🔢 Number of epochs (default 100): ").strip()
            epochs = int(epochs) if epochs else 100
            
            batch_size = input("🔢 Batch size (default 128): ").strip()
            batch_size = int(batch_size) if batch_size else 128
            
            lr = input("🔢 Learning rate (default 0.001): ").strip()
            lr = float(lr) if lr else 0.001
            
            # Train
            print(f"\n🧠 Training Neural Network...")
            print(f"   Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
            
            nn_model = FoodWasteNeuralNetwork(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr
            )
            
            nn_model.fit(X_train, y_train)
            
            # Evaluate
            predictions = nn_model.predict(X_test)
            mae = np.mean(np.abs(y_test.values - predictions))
            rmse = np.sqrt(np.mean((y_test.values - predictions)**2))
            
            print(f"\n✅ Training Complete!")
            print(f"   MAE: {mae:.2f} kg")
            print(f"   RMSE: {rmse:.2f} kg")
            
            # Save
            save = input("\n💾 Save this neural network? (y/n): ").strip().lower()
            if save == 'y':
                joblib.dump(nn_model, 'data/models/neural_network.pkl')
                joblib.dump(preprocessor, 'data/models/nn_preprocessor.pkl')
                print("✅ Saved to data/models/neural_network.pkl")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== WORKFLOW 10: MANAGE CONFIGS ====================
    def manage_configs(self):
        """Manage restaurant configurations"""
        print("\n" + "="*70)
        print("💾 MANAGE RESTAURANT CONFIGURATIONS")
        print("="*70)
        
        config_manager = RestaurantConfigManager()
        configs = config_manager.list_saved_configs()
        
        if not configs:
            print("\n📋 No saved configurations found.")
            return
        
        print(f"\n📋 Found {len(configs)} saved configurations:")
        for i, name in enumerate(configs, 1):
            print(f"   {i}. {name}")
        
        print("\nOptions:")
        print("   1. View configuration details")
        print("   2. Delete configuration")
        print("   0. Back to main menu")
        
        choice = input("\n🔢 Select option: ").strip()
        
        if choice == '1':
            idx = input("Enter configuration number: ").strip()
            try:
                restaurant = configs[int(idx) - 1]
                config = config_manager.load_restaurant_config(restaurant)
                print(f"\n📋 Configuration for {restaurant}:")
                print(json.dumps(config, indent=2))
            except:
                print("❌ Invalid selection")
        
        elif choice == '2':
            idx = input("Enter configuration number to delete: ").strip()
            try:
                restaurant = configs[int(idx) - 1]
                confirm = input(f"⚠️ Delete config for {restaurant}? (y/n): ").strip().lower()
                if confirm == 'y':
                    filepath = f'configs/restaurant_formats/{restaurant}_config.json'
                    os.remove(filepath)
                    print(f"✅ Deleted configuration for {restaurant}")
            except:
                print("❌ Invalid selection or deletion failed")
    
    # ==================== WORKFLOW 11: DOCUMENTATION ====================
    def show_documentation(self):
        """Show system documentation"""
        print("\n" + "="*70)
        print("📖 SYSTEM DOCUMENTATION")
        print("="*70)
        
        doc = """
        
🍔 FOOD WASTE PREDICTION SYSTEM - COMPREHENSIVE GUIDE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 PROJECT STRUCTURE:
   src/
      ├── data_preprocessing.py      - Data cleaning & feature engineering
      ├── model_training.py          - CPU model training
      ├── model_training_gpu.py      - GPU-accelerated training
      ├── neural_network_gpu.py      - PyTorch NN wrapper
      ├── neural_network_waste_gpu.py - Food waste specific NN
      ├── evaluation.py              - CPU model evaluation
      ├── evaluation_gpu.py          - GPU model evaluation
      ├── prediction.py              - Prediction interface
      ├── future_prediction.py       - Future predictions module
      ├── universal_data_adapter.py  - Universal CSV adapter
      └── restaurant_config_manager.py - Config management

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 QUICK START GUIDE:

1. First Time Setup:
   - Run Option 1 (Quick Start) to generate sample data and train

2. Using Your Own Data:
   - Standard format? → Option 2
   - Any format? → Option 3 (Universal Adapter)

3. Making Predictions:
   - Future predictions → Option 5
   - Real-time → Option 6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DATA FORMAT:

Standard columns (required):
   - date, food_item, quantity_wasted

Recommended columns:
   - food_category, quantity_prepared, quantity_sold
   - customer_count, weather, temperature
   - day_of_week, special_event

Universal Adapter handles ANY format!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 MODEL OPTIONS:

CPU Models:
   - Linear/Ridge/Lasso Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost (CPU)

GPU Models:
   - XGBoost (GPU)
   - LightGBM (GPU)
   - Neural Network (PyTorch)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIPS:

- Use GPU for datasets > 10,000 rows
- Universal Adapter uses fuzzy matching (70% threshold)
- Save restaurant configs to avoid re-mapping
- Neural networks need more data (>5,000 rows recommended)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 SUPPORT:
   Check README.md for detailed documentation
   
        """
        
        print(doc)
        input("\nPress Enter to continue...")
    
    # ==================== MAIN LOOP ====================
    def run(self):
        """Main execution loop"""
        while True:
            self.display_menu()
            
            choice = input("\n🔢 Enter your choice (0-11): ").strip()
            
            if choice == '0':
                print("\n👋 Thank you for using Food Waste Prediction System!")
                if self.gpu_available:
                    torch.cuda.empty_cache()
                break
            
            elif choice == '1':
                self.quick_start()
            
            elif choice == '2':
                self.train_standard_format()
            
            elif choice == '3':
                self.train_any_format()
            
            elif choice == '4':
                self.evaluate_models()
            
            elif choice == '5':
                self.future_predictions()
            
            elif choice == '6':
                self.realtime_predictions()
            
            elif choice == '7':
                self.run_benchmark()
            
            elif choice == '8':
                self.model_comparison()
            
            elif choice == '9':
                self.train_neural_network_only()
            
            elif choice == '10':
                self.manage_configs()
            
            elif choice == '11':
                self.show_documentation()
            
            else:
                print("\n❌ Invalid choice. Please try again.")
            
            # Cleanup
            if self.gpu_available:
                torch.cuda.empty_cache()


# ==================== COMMAND LINE INTERFACE ====================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Food Waste Prediction System - Unified Pipeline'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'train', 'adapt', 'evaluate', 'predict', 'future', 'benchmark', 'menu'],
        default='menu',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV data file'
    )
    
    parser.add_argument(
        '--restaurant',
        type=str,
        default='default',
        help='Restaurant name (for adapt mode)'
    )
    
    parser.add_argument(
        '--gpu',
        type=bool,
        default=True,
        help='Use GPU acceleration'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs for neural network'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for neural network'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    pipeline = FoodWastePipeline()
    
    if args.mode == 'menu':
        # Interactive menu mode
        pipeline.run()
    
    elif args.mode == 'quick':
        # Quick start mode
        pipeline.quick_start()
    
    elif args.mode == 'train':
        # Train with standard format
        if not args.data:
            print("❌ --data argument required for train mode")
            return
        pipeline.train_standard_format()
    
    elif args.mode == 'adapt':
        # Train with universal adapter
        if not args.data:
            print("❌ --data argument required for adapt mode")
            return
        pipeline.restaurant_name = args.restaurant
        pipeline.train_any_format()
    
    elif args.mode == 'evaluate':
        # Evaluate existing models
        pipeline.evaluate_models()
    
    elif args.mode == 'predict':
        # Real-time predictions
        pipeline.realtime_predictions()
    
    elif args.mode == 'future':
        # Future predictions
        pipeline.future_predictions()
    
    elif args.mode == 'benchmark':
        # GPU benchmark
        pipeline.run_benchmark()


if __name__ == "__main__":
    main()