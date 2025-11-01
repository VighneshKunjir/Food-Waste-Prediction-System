# main.py
"""
Food Waste Prediction System - Main Pipeline
Complete orchestrator with menu-driven interface
"""

import os
import sys
from datetime import datetime

# Workflows
from workflows.training_workflow import TrainingWorkflow
from workflows.prediction_workflow import PredictionWorkflow
from workflows.evaluation_workflow import EvaluationWorkflow
from workflows.benchmark_workflow import BenchmarkWorkflow

# Storage
from storage.restaurant_manager import RestaurantManager
from storage.prediction_storage import PredictionStorage

# Core
from core.data_loader import DataLoader

# Utils
from utils.gpu_utils import GPUManager
from utils.file_utils import FileManager
from utils.logger import Logger


class FoodWastePredictionSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        """Initialize the system"""
        self.logger = Logger(name='MainSystem')
        self.gpu_manager = GPUManager()
        self.restaurant_manager = RestaurantManager()
        self.prediction_storage = PredictionStorage()
        
        # Create project structure
        FileManager.create_project_structure()
        
        # Current context
        self.current_restaurant = None
        self.current_workflow = None
        
        self.logger.info("System initialized")
    
    def display_header(self):
        """Display system header"""
        print("\n" + "="*80)
        print("🍔 FOOD WASTE PREDICTION SYSTEM".center(80))
        print("AI-Powered Waste Reduction for Restaurants".center(80))
        print("="*80)
        
        # GPU Status
        if self.gpu_manager.gpu_available:
            print(f"🔥 GPU: {self.gpu_manager.device_name}".center(80))
        else:
            print("💻 Running on CPU".center(80))
        
        print("="*80)
    
    def display_main_menu(self):
        """Display main menu"""
        print("\n" + "="*80)
        print("📋 MAIN MENU")
        print("="*80)
        
        print("\n1️⃣  TRAIN NEW MODEL")
        print("     1.1 - Generate Sample Data & Train")
        print("     1.2 - Use Custom CSV Data")
        
        print("\n2️⃣  MAKE PREDICTIONS")
        print("     2.1 - Future Waste Prediction")
        print("     2.2 - Real-time Prediction")
        print("     2.3 - Batch Prediction")
        
        print("\n3️⃣  MANAGE RESTAURANTS")
        print("     3.1 - View Restaurant Info")
        print("     3.2 - View Predictions History")
        print("     3.3 - Manage Configurations")
        print("     3.4 - Export Restaurant Data")
        
        print("\n4️⃣  MODEL EVALUATION")
        print("     4.1 - Evaluate Best Model")
        print("     4.2 - Compare All Models")
        print("     4.3 - Generate Comprehensive Report")
        
        print("\n5️⃣  BENCHMARK GPU PERFORMANCE")
        
        print("\n0️⃣  EXIT")
        
        print("\n" + "="*80)
    
    def run(self):
        """Main execution loop"""
        self.display_header()
        
        while True:
            self.display_main_menu()
            
            choice = input("\n🔢 Enter your choice: ").strip()
            
            if choice == '0':
                self.exit_system()
                break
            
            elif choice == '1':
                self.training_menu()
            
            elif choice == '2':
                self.prediction_menu()
            
            elif choice == '3':
                self.restaurant_management_menu()
            
            elif choice == '4':
                self.evaluation_menu()
            
            elif choice == '5':
                self.run_benchmark()
            
            else:
                print("\n❌ Invalid choice. Please try again.")
    
    # ==================== 1. TRAINING MENU ====================
    
    def training_menu(self):
        """Training submenu"""
        while True:
            print("\n" + "="*80)
            print("1️⃣  TRAINING MENU")
            print("="*80)
            
            print("\n1.1 - Generate Sample Data & Train")
            print("1.2 - Use Custom CSV Data")
            print("0   - Back to Main Menu")
            
            choice = input("\n🔢 Enter choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '1.1':
                self.train_with_sample_data()
            elif choice == '1.2':
                self.train_with_custom_csv()
            else:
                print("❌ Invalid choice")
    
    def train_with_sample_data(self):
        """Train with generated sample data"""
        print("\n" + "="*80)
        print("1.1 - TRAIN WITH SAMPLE DATA")
        print("="*80)
        
        # Get restaurant name
        restaurant_name = input("\n🏪 Enter restaurant name: ").strip()
        if not restaurant_name:
            print("❌ Restaurant name required")
            return
        
        # Get parameters
        n_days = input("📅 Number of days to generate (default 365): ").strip()
        n_days = int(n_days) if n_days else 365
        
        # GPU and Neural Network options
        use_gpu, include_neural_network, nn_params = self._ask_gpu_option()

        # Initialize workflow
        workflow = TrainingWorkflow(
            restaurant_name, 
            use_gpu=use_gpu,
            include_neural_network=include_neural_network,
            nn_params=nn_params  #  NEW
        )
        
        # Train
        result = workflow.train_with_sample_data(n_days=n_days)
        
        if result['success']:
            print("\n✅ Training completed successfully!")
            self.current_restaurant = restaurant_name
            
            # Ask for next action
            self._post_training_menu(restaurant_name)
        else:
            print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
    
    def train_with_custom_csv(self):
        """Train with custom CSV data"""
        print("\n" + "="*80)
        print("1.2 - TRAIN WITH CUSTOM CSV")
        print("="*80)
        
        # Get restaurant name
        restaurant_name = input("\n🏪 Enter restaurant name: ").strip()
        if not restaurant_name:
            print("❌ Restaurant name required")
            return
        
        # Get CSV path
        csv_path = input("📁 Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return
        
        # Choose training mode
        print("\n🔧 Training Mode:")
        print("1 - CPU Training")
        print("2 - GPU Training")

        mode = input("\nChoose mode (1/2): ").strip()

        use_gpu = (mode == '2')
        include_neural_network = False
        nn_params = None

        if use_gpu and self.gpu_manager.gpu_available:
            # Ask about neural network for GPU mode
            print("\n" + "─"*60)
            print("🧠 NEURAL NETWORK TRAINING")
            print("─"*60)
            nn_choice = input("   Train with Neural Network? (y/n): ").strip().lower()
            include_neural_network = (nn_choice == 'y')
            
            if include_neural_network:
                print("   ✅ Will train with Neural Network")
                # Ask for parameters
                nn_params = self._ask_nn_parameters()
            else:
                print("   ⏭️  Neural Network will be skipped")

        # Initialize workflow
        workflow = TrainingWorkflow(
            restaurant_name, 
            use_gpu=use_gpu,
            include_neural_network=include_neural_network,
            nn_params=nn_params  # ⭐ NEW
        )
        
        # Train
        result = workflow.train_with_custom_csv(csv_path)
        
        if result['success']:
            print("\n✅ Training completed successfully!")
            self.current_restaurant = restaurant_name
            
            # Ask for next action
            self._post_training_menu(restaurant_name)
        else:
            print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
    
    def _post_training_menu(self, restaurant_name):
        """Menu after successful training"""
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE - NEXT STEPS")
        print("="*80)
        
        print("\n1 - Make Predictions Now")
        print("2 - Evaluate Model")
        print("3 - Return to Main Menu")
        
        choice = input("\n🔢 Choose next action: ").strip()
        
        if choice == '1':
            self.current_restaurant = restaurant_name
            self.prediction_menu()
        elif choice == '2':
            self.current_restaurant = restaurant_name
            self.evaluation_menu()
    
    # ==================== 2. PREDICTION MENU ====================
    
    def prediction_menu(self):
        """Prediction submenu"""
        # Select restaurant if not set
        if not self.current_restaurant:
            self.current_restaurant = self._select_restaurant()
            if not self.current_restaurant:
                return
        
        while True:
            print("\n" + "="*80)
            print(f"2️⃣  PREDICTION MENU - {self.current_restaurant}")
            print("="*80)
            
            print("\n2.1 - Future Waste Prediction")
            print("2.2 - Real-time Prediction (Interactive)")
            print("2.3 - Batch Prediction from CSV")
            print("9   - Change Restaurant")
            print("0   - Back to Main Menu")
            
            choice = input("\n🔢 Enter choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '9':
                self.current_restaurant = self._select_restaurant()
            elif choice == '2.1':
                self.future_prediction()
            elif choice == '2.2':
                self.realtime_prediction()
            elif choice == '2.3':
                self.batch_prediction()
            else:
                print("❌ Invalid choice")
    
    def future_prediction(self):
        """Future waste prediction"""
        print("\n" + "="*80)
        print("2.1 - FUTURE WASTE PREDICTION")
        print("="*80)
        
        print("\n1 - Single Day Prediction")
        print("2 - Multiple Items (Single Day)")
        print("3 - Weekly Forecast (7 Days)")
        
        choice = input("\n🔢 Choose prediction type: ").strip()
        
        workflow = PredictionWorkflow(self.current_restaurant)
        
        if choice == '1':
            # Single day prediction
            date = input("\n📅 Enter date (YYYY-MM-DD): ").strip()
            food_item = input("🍔 Enter food item: ").strip()
            
            customers = input("👥 Expected customers (or Enter to auto-estimate): ").strip()
            customers = int(customers) if customers else None
            
            workflow.run_future_prediction(
                prediction_type='single',
                date=date,
                food_item=food_item,
                expected_customers=customers
            )
        
        elif choice == '2':
            # Multiple items
            date = input("\n📅 Enter date (YYYY-MM-DD): ").strip()
            
            workflow.run_future_prediction(
                prediction_type='multiple',
                date=date
            )
        
        elif choice == '3':
            # Weekly forecast
            start_date = input("\n📅 Start date (YYYY-MM-DD) or Enter for tomorrow: ").strip()
            start_date = start_date if start_date else None
            
            workflow.run_future_prediction(
                prediction_type='weekly',
                start_date=start_date
            )
        
        print("\n✅ Prediction saved to restaurant predictions folder")
    
    def realtime_prediction(self):
        """Real-time interactive prediction"""
        print("\n" + "="*80)
        print("2.2 - REAL-TIME PREDICTION")
        print("="*80)
        
        workflow = PredictionWorkflow(self.current_restaurant)
        workflow.run_realtime_prediction(interactive=True)
    
    def batch_prediction(self):
        """Batch prediction from CSV"""
        print("\n" + "="*80)
        print("2.3 - BATCH PREDICTION")
        print("="*80)
        
        csv_path = input("\n📁 Enter CSV file path: ").strip().strip('"')
        
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return
        
        workflow = PredictionWorkflow(self.current_restaurant)
        result = workflow.run_batch_prediction(csv_path)
        
        if result is not None:
            print("\n✅ Batch prediction completed!")
            print(f"📊 Total records: {len(result)}")
    
    # ==================== 3. RESTAURANT MANAGEMENT MENU ====================
    
    def restaurant_management_menu(self):
        """Restaurant management submenu"""
        while True:
            print("\n" + "="*80)
            print("3️⃣  RESTAURANT MANAGEMENT")
            print("="*80)
            
            print("\n3.1 - View Restaurant Information")
            print("3.2 - View Predictions History")
            print("3.3 - Manage Configurations")
            print("3.4 - Export Restaurant Data")
            print("3.5 - List All Restaurants")
            print("3.6 - Delete Restaurant")
            print("0   - Back to Main Menu")
            
            choice = input("\n🔢 Enter choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '3.1':
                self.view_restaurant_info()
            elif choice == '3.2':
                self.view_predictions_history()
            elif choice == '3.3':
                self.manage_configurations()
            elif choice == '3.4':
                self.export_restaurant_data()
            elif choice == '3.5':
                self.list_all_restaurants()
            elif choice == '3.6':
                self.delete_restaurant()
            else:
                print("❌ Invalid choice")
    
    def view_restaurant_info(self):
        """View restaurant information"""
        restaurant = self._select_restaurant()
        if not restaurant:
            return
        
        self.restaurant_manager.display_restaurant_info(restaurant)
        
        # Show statistics
        stats = self.restaurant_manager.get_statistics(restaurant)
        
        if stats['realtime_stats']:
            print("\n📊 Real-time Prediction Statistics:")
            for key, value in stats['realtime_stats'].items():
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
        
        if stats['future_stats']:
            print("\n🔮 Future Prediction Statistics:")
            for key, value in stats['future_stats'].items():
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
    
    def view_predictions_history(self):
        """View predictions history"""
        restaurant = self._select_restaurant()
        if not restaurant:
            return
        
        print("\n" + "="*80)
        print(f"📜 PREDICTIONS HISTORY - {restaurant}")
        print("="*80)
        
        print("\n1 - Real-time Predictions")
        print("2 - Future Predictions")
        print("3 - Both")
        
        choice = input("\n🔢 Choose: ").strip()
        
        workflow = PredictionWorkflow(restaurant)
        
        if choice in ['1', '3']:
            print("\n📱 Real-time Predictions:")
            workflow.view_prediction_history('realtime', limit=10)
        
        if choice in ['2', '3']:
            print("\n🔮 Future Predictions:")
            workflow.view_prediction_history('future', limit=10)
    
    def manage_configurations(self):
        """Manage restaurant configurations"""
        restaurant = self._select_restaurant()
        if not restaurant:
            return
        
        config = self.restaurant_manager.get_restaurant_config(restaurant)
        
        if config:
            print("\n📋 Current Configuration:")
            import json
            print(json.dumps(config, indent=2))
        
        print("\n1 - Update Configuration")
        print("2 - View Only")
        
        choice = input("\n🔢 Choose: ").strip()
        
        if choice == '1':
            key = input("\nEnter config key to update: ").strip()
            value = input("Enter new value: ").strip()
            
            self.restaurant_manager.update_restaurant_config(
                restaurant,
                {key: value}
            )
            
            print("✅ Configuration updated")
    
    def export_restaurant_data(self):
        """Export restaurant data"""
        restaurant = self._select_restaurant()
        if not restaurant:
            return
        
        output_dir = self.restaurant_manager.export_restaurant_data(restaurant)
        print(f"\n✅ Data exported to: {output_dir}")
    
    def list_all_restaurants(self):
        """List all restaurants"""
        restaurants = self.restaurant_manager.list_restaurants()
        
        print("\n" + "="*80)
        print("🏪 ALL RESTAURANTS")
        print("="*80)
        
        if not restaurants:
            print("\n⚠️ No restaurants found")
            return
        
        print(f"\nTotal: {len(restaurants)} restaurants\n")
        
        for i, restaurant in enumerate(restaurants, 1):
            info = self.restaurant_manager.get_restaurant_info(restaurant)
            
            print(f"{i}. {restaurant}")
            
            if info and info['metadata']:
                print(f"   Created: {info['metadata'].get('created_date', 'N/A')}")
                print(f"   Models: {info['metadata'].get('total_models', 0)}")
                print(f"   Predictions: {info['metadata'].get('total_predictions', 0)}")
            
            print()
    
    def delete_restaurant(self):
        """Delete a restaurant"""
        restaurant = self._select_restaurant()
        if not restaurant:
            return
        
        success = self.restaurant_manager.delete_restaurant(restaurant, confirm=True)
        
        if success and self.current_restaurant == restaurant:
            self.current_restaurant = None
    
    # ==================== 4. EVALUATION MENU ====================
    
    def evaluation_menu(self):
        """Evaluation submenu"""
        # Select restaurant if not set
        if not self.current_restaurant:
            self.current_restaurant = self._select_restaurant()
            if not self.current_restaurant:
                return
        
        while True:
            print("\n" + "="*80)
            print(f"4️⃣  MODEL EVALUATION - {self.current_restaurant}")
            print("="*80)
            
            print("\n4.1 - Evaluate Best Model")
            print("4.2 - Compare All Models")
            print("4.3 - Generate Comprehensive Report")
            print("9   - Change Restaurant")
            print("0   - Back to Main Menu")
            
            choice = input("\n🔢 Enter choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '9':
                self.current_restaurant = self._select_restaurant()
            elif choice == '4.1':
                self.evaluate_best_model()
            elif choice == '4.2':
                self.compare_all_models()
            elif choice == '4.3':
                self.generate_comprehensive_report()
            else:
                print("❌ Invalid choice")
    
    def evaluate_best_model(self):
        """Evaluate best model"""
        print("\n" + "="*80)
        print("4.1 - EVALUATE BEST MODEL")
        print("="*80)
        
        # Find evaluation data
        test_data_path = self._find_evaluation_data()
        
        if not test_data_path:
            return
        
        try:
            # Load data
            loader = DataLoader()
            df = loader.load_csv(test_data_path)
            
            # Check for required column
            if 'quantity_wasted' not in df.columns:
                df = self._map_waste_column(df)
                if df is None:
                    return
            
            # Load preprocessor
            from core.preprocessor import Preprocessor
            from storage.model_storage import ModelStorage
            
            model_storage = ModelStorage()
            preprocessor = model_storage.load_preprocessor(self.current_restaurant)
            
            X_test = preprocessor.transform(df)
            y_test = df['quantity_wasted']
            
            # Evaluate
            workflow = EvaluationWorkflow(self.current_restaurant)
            metrics = workflow.evaluate_best_model(X_test, y_test, generate_plots=True)
            
            print("\n✅ Evaluation complete!")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    
    def compare_all_models(self):
        """Compare all models"""
        print("\n" + "="*80)
        print("4.2 - COMPARE ALL MODELS")
        print("="*80)
        
        # Find evaluation data
        test_data_path = self._find_evaluation_data()
        
        if not test_data_path:
            return
        
        try:
            # Load data
            loader = DataLoader()
            df = loader.load_csv(test_data_path)
            
            # Check for quantity_wasted column
            if 'quantity_wasted' not in df.columns:
                df = self._map_waste_column(df)
                if df is None:
                    return
            
            # Proceed with comparison
            from core.preprocessor import Preprocessor
            from storage.model_storage import ModelStorage
            
            model_storage = ModelStorage()
            preprocessor = model_storage.load_preprocessor(self.current_restaurant)
            
            print("\n🔄 Preprocessing data...")
            X_test = preprocessor.transform(df)
            y_test = df['quantity_wasted']
            
            print(f"✅ Prepared {len(X_test)} samples for evaluation")
            
            # Compare
            workflow = EvaluationWorkflow(self.current_restaurant)
            comparison = workflow.compare_all_models(X_test, y_test)
            
            print("\n✅ Comparison complete!")
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*80)
        print("4.3 - GENERATE COMPREHENSIVE REPORT")
        print("="*80)
        
        # Find evaluation data
        test_data_path = self._find_evaluation_data()
        
        if not test_data_path:
            return
        
        try:
            # Load data
            loader = DataLoader()
            df = loader.load_csv(test_data_path)
            
            # Check for required column
            if 'quantity_wasted' not in df.columns:
                df = self._map_waste_column(df)
                if df is None:
                    return
            
            # Load preprocessor
            from core.preprocessor import Preprocessor
            from storage.model_storage import ModelStorage
            
            model_storage = ModelStorage()
            preprocessor = model_storage.load_preprocessor(self.current_restaurant)
            
            X_test = preprocessor.transform(df)
            y_test = df['quantity_wasted']
            
            # Generate report
            workflow = EvaluationWorkflow(self.current_restaurant)
            report_dir = workflow.generate_comprehensive_report(X_test, y_test)
            
            if report_dir:
                print(f"\n✅ Report generated: {report_dir}")
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== 5. BENCHMARK ====================
    
    def run_benchmark(self):
        """Run GPU benchmark"""
        print("\n" + "="*80)
        print("5️⃣  GPU PERFORMANCE BENCHMARK")
        print("="*80)
        
        print("\n1 - Quick GPU Check")
        print("2 - Full Benchmark (Multiple Data Sizes)")
        
        choice = input("\n🔢 Choose: ").strip()
        
        workflow = BenchmarkWorkflow()
        
        if choice == '1':
            workflow.quick_gpu_check()
        
        elif choice == '2':
            print("\n⚠️ Full benchmark may take several minutes...")
            proceed = input("Continue? (y/n): ").strip().lower()
            
            if proceed == 'y':
                # Get data sizes
                sizes_input = input("\nData sizes (comma-separated, or Enter for default): ").strip()
                
                if sizes_input:
                    sizes = [int(s.strip()) for s in sizes_input.split(',')]
                else:
                    sizes = [1000, 5000, 10000, 25000]
                
                results = workflow.run_complete_benchmark(data_sizes=sizes)
                
                print("\n✅ Benchmark complete!")
    
    # ==================== HELPER METHODS ====================
    
    def _select_restaurant(self):
        """Select restaurant from list"""
        restaurants = self.restaurant_manager.list_restaurants()
        
        if not restaurants:
            print("\n⚠️ No restaurants found. Please train a model first.")
            return None
        
        print("\n" + "="*80)
        print("🏪 SELECT RESTAURANT")
        print("="*80)
        
        for i, restaurant in enumerate(restaurants, 1):
            print(f"{i}. {restaurant}")
        
        choice = input("\n🔢 Enter number: ").strip()
        
        try:
            idx = int(choice) - 1
            return restaurants[idx]
        except:
            print("❌ Invalid selection")
            return None
    
    def _ask_gpu_option(self):
        """
        Ask user if they want to use GPU and neural network.
        Neural Network prompt appears ONLY for GPU mode.
        
        Returns:
            Tuple of (use_gpu, include_neural_network, nn_params)
        """
        if not self.gpu_manager.gpu_available:
            print("\n💻 GPU not available, using CPU")
            return False, False, None
        
        print("\n🔥 GPU available!")
        print(f"   Device: {self.gpu_manager.device_name}")
        
        choice = input("\nUse GPU acceleration? (y/n): ").strip().lower()
        
        use_gpu = (choice == 'y')
        include_neural_network = False
        nn_params = None
        
        if use_gpu:
            # Ask about neural network ONLY when GPU is selected
            print("\n" + "─"*60)
            print("🧠 NEURAL NETWORK TRAINING")
            print("─"*60)
            print("   Neural networks can provide high accuracy but:")
            print("   ✓ Take longer to train (5-10x slower)")
            print("   ✓ Require more GPU memory")
            print("   ✓ May not always be the best model")
            print()
            print("   Other GPU models (XGBoost, LightGBM, CatBoost) are often")
            print("   equally accurate and much faster to train.")
            
            nn_choice = input("\n   Train with Neural Network? (y/n): ").strip().lower()
            include_neural_network = (nn_choice == 'y')
            
            if include_neural_network:
                print("   ✅ Neural Network will be included")
                
                # Ask for parameter customization
                nn_params = self._ask_nn_parameters()
            else:
                print("   ⏭️  Neural Network will be skipped")
                print("   ℹ️  Will train with 10 other models")
        
        return use_gpu, include_neural_network, nn_params

    def _ask_nn_parameters(self):
        """
        Ask user for neural network parameters
        
        Returns:
            Dictionary of NN parameters or None for defaults
        """
        print("\n" + "─"*60)
        print("⚙️  NEURAL NETWORK PARAMETERS")
        print("─"*60)
        print("\n1 - Use Default Parameters (Recommended)")
        print("    • Epochs: 100")
        print("    • Batch Size: 128")
        print("    • Learning Rate: 0.001")
        print()
        print("2 - Custom Parameters (Advanced)")
        print("    • Fine-tune for your specific data")
        
        param_choice = input("\nChoose option (1/2): ").strip()
        
        if param_choice == '2':
            print("\n📝 Enter custom parameters (press Enter for default):")
            
            try:
                # Epochs
                epochs_input = input(f"   Epochs [default: 100]: ").strip()
                epochs = int(epochs_input) if epochs_input else 100
                
                # Batch Size
                batch_input = input(f"   Batch Size [default: 128]: ").strip()
                batch_size = int(batch_input) if batch_input else 128
                
                # Learning Rate
                lr_input = input(f"   Learning Rate [default: 0.001]: ").strip()
                learning_rate = float(lr_input) if lr_input else 0.001
                
                # Hidden Layers (optional advanced feature)
                print("\n   Advanced (optional):")
                hidden_input = input(f"   Hidden Layer Sizes [default: 256,128,64,32]: ").strip()
                if hidden_input:
                    hidden_layers = [int(x.strip()) for x in hidden_input.split(',')]
                else:
                    hidden_layers = [256, 128, 64, 32]
                
                # Dropout (optional)
                dropout_input = input(f"   Dropout Rate [default: 0.3]: ").strip()
                dropout = float(dropout_input) if dropout_input else 0.3
                
                nn_params = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_layers': hidden_layers,
                    'dropout': dropout
                }
                
                # Display summary
                print("\n✅ Custom parameters:")
                print(f"   Epochs: {epochs}")
                print(f"   Batch Size: {batch_size}")
                print(f"   Learning Rate: {learning_rate}")
                print(f"   Hidden Layers: {hidden_layers}")
                print(f"   Dropout: {dropout}")
                
                confirm = input("\nConfirm these parameters? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    return nn_params
                else:
                    print("   Using default parameters instead")
                    return None
                    
            except ValueError as e:
                print(f"\n⚠️  Invalid input: {e}")
                print("   Using default parameters instead")
                return None
        else:
            print("\n✅ Using default parameters")
            return None
    
    def exit_system(self):
        """Exit the system"""
        print("\n" + "="*80)
        print("👋 THANK YOU FOR USING FOOD WASTE PREDICTION SYSTEM")
        print("="*80)
        
        # Cleanup
        if self.gpu_manager.gpu_available:
            self.gpu_manager.clear_cache()
        
        self.logger.info("System shutdown")
        
        print("\n🌍 Together, we can reduce food waste and help the planet!")
        print("\n")

    def _find_evaluation_data(self):
        """Helper method to find available evaluation data"""
        print("\n🔍 Searching for available data files...")
        
        available_files = []
        
        # Possible data locations
        possible_paths = [
            f'data/restaurants/{self.current_restaurant}/sample_data.csv',
            'data/raw/adapted_data.csv',
            'data/processed/processed_data.csv',
            'data/raw/food_waste_data.csv',
        ]
        
        # Check restaurant directory
        restaurant_dir = f'data/restaurants/{self.current_restaurant}'
        if os.path.exists(restaurant_dir):
            for file in os.listdir(restaurant_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(restaurant_dir, file)
                    if file_path not in possible_paths:
                        possible_paths.append(file_path)
        
        # Check which files exist
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df_check = pd.read_csv(path)
                    has_waste = 'quantity_wasted' in df_check.columns
                    available_files.append({
                        'path': path,
                        'rows': len(df_check),
                        'has_waste': has_waste,
                        'columns': len(df_check.columns)
                    })
                except:
                    pass
        
        if not available_files:
            print("❌ No data files found")
            print("\n💡 Please enter the path to your training data CSV")
            test_data_path = input("\n📁 Enter CSV path: ").strip().strip('"')
            return test_data_path if os.path.exists(test_data_path) else None
        
        print(f"\n📁 Found {len(available_files)} data file(s):\n")
        
        for i, file_info in enumerate(available_files, 1):
            status = "✅" if file_info['has_waste'] else "⚠️"
            print(f"{i}. {status} {file_info['path']}")
            print(f"   Rows: {file_info['rows']:,} | Columns: {file_info['columns']} | Has waste: {file_info['has_waste']}")
        
        print(f"\n0. Enter custom path")
        
        choice = input(f"\n🔢 Select file (0-{len(available_files)}): ").strip()
        
        if choice == '0':
            test_data_path = input("\n📁 Enter CSV path: ").strip().strip('"')
        else:
            try:
                file_idx = int(choice) - 1
                test_data_path = available_files[file_idx]['path']
                print(f"\n✅ Selected: {test_data_path}")
            except:
                print("❌ Invalid selection")
                return None
        
        return test_data_path if os.path.exists(test_data_path) else None
    
    def _map_waste_column(self, df):
        """Helper method to map waste column if not found"""
        print("\n❌ 'quantity_wasted' column not found")
        print(f"\n📋 Available columns:")
        
        for i, col in enumerate(df.columns, 1):
            sample = df[col].iloc[0] if len(df) > 0 else "N/A"
            print(f"   {i}. {col:30} (sample: {sample})")
        
        print("\n💡 Which column contains waste/wastage data?")
        choice = input("🔢 Enter column number (or 0 to cancel): ").strip()
        
        if choice and choice != '0':
            try:
                col_idx = int(choice) - 1
                waste_col = df.columns[col_idx]
                print(f"\n✅ Using '{waste_col}' as quantity_wasted")
                df['quantity_wasted'] = df[waste_col]
                return df
            except:
                print("❌ Invalid selection")
                return None
        else:
            return None



def main():
    """Main entry point"""
    try:
        system = FoodWastePredictionSystem()
        system.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ System interrupted by user")
        print("👋 Goodbye!")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()