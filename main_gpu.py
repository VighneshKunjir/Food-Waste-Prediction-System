# main_gpu.py
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model_training_gpu import ModelTrainerGPU
from src.neural_network_waste_gpu import FoodWasteNeuralNetwork
from src.evaluation_gpu import FoodWasteEvaluator

def main():
    """Main pipeline for GPU-accelerated Food Waste Prediction"""
    
    print("\n" + "="*60)
    print("🍔 FOOD WASTE PREDICTION SYSTEM - GPU ACCELERATED")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()
    else:
        print("\n⚠️ No GPU detected. Some models will run on CPU.")
    
    # Step 1: Data Generation
    print("\n1️⃣ DATA GENERATION")
    print("-"*40)
    
    if not os.path.exists('data/raw/food_waste_data.csv'):
        print("📊 Generating sample data...")
        from generate_sample_data import generate_sample_data
        generate_sample_data(365)
    else:
        print("✅ Data file found")
    
    # Step 2: Data Preprocessing
    print("\n2️⃣ DATA PREPROCESSING FOR WASTE PREDICTION")
    print("-"*40)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, processed_df = preprocessor.prepare_data(
        'data/raw/food_waste_data.csv'
    )
    
    print(f"\n📊 Target Variable (Waste) Statistics:")
    print(f"   Training samples: {len(y_train)}")
    print(f"   Mean waste: {y_train.mean():.2f} kg")
    print(f"   Median waste: {y_train.median():.2f} kg")
    print(f"   Std deviation: {y_train.std():.2f} kg")
    
    # Step 3: Traditional + GPU Models Training
    print("\n3️⃣ TRAINING WASTE PREDICTION MODELS")
    print("-"*40)
    
    trainer = ModelTrainerGPU()
    results = trainer.train_all_models(X_train, y_train)
    
    # Step 4: Neural Network Training (GPU)
    if torch.cuda.is_available():
        print("\n4️⃣ NEURAL NETWORK FOR WASTE PREDICTION (GPU)")
        print("-"*40)
        
        try:
            nn_model = FoodWasteNeuralNetwork(epochs=100, batch_size=128)
            nn_model.fit(X_train, y_train)
            
            # Quick evaluation
            nn_pred = nn_model.predict(X_test[:100])
            nn_mae = np.mean(np.abs(y_test[:100].values - nn_pred))
            
            print(f"\n✅ Neural Network trained successfully")
            print(f"   Sample MAE: {nn_mae:.2f} kg")
            
            # Add to results
            results['Neural Network (GPU)'] = {
                'model': nn_model,
                'mae': nn_mae,
                'type': 'GPU'
            }
            
        except Exception as e:
            print(f"   ⚠️ Neural Network training failed: {e}")
    
    # Step 5: Comprehensive Evaluation with Visualizations
    print("\n5️⃣ COMPREHENSIVE MODEL EVALUATION")
    print("-"*40)
    
    evaluator = FoodWasteEvaluator()
    
    # Create directory for plots
    os.makedirs('data/plots', exist_ok=True)
    
    # Evaluate each model
    for model_name, model_data in results.items():
        try:
            model = model_data['model']
            metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)
            
            # Create individual plot for each model
            plot_path = f'data/plots/{model_name.replace(" ", "_")}_analysis.png'
            evaluator.plot_model_performance(model_name, save_path=plot_path)
            
        except Exception as e:
            print(f"   ⚠️ Failed to evaluate {model_name}: {e}")
    
    # Step 6: Model Comparison
    print("\n6️⃣ MODEL COMPARISON")
    print("-"*40)
    
    comparison_plot_path = 'data/plots/models_comparison.png'
    evaluator.compare_all_models(save_path=comparison_plot_path)
    
    # Step 7: Save Best Model
    if trainer.best_model:
        import joblib
        os.makedirs('data/models', exist_ok=True)
        
        # Save best model
        joblib.dump(trainer.best_model, 'data/models/best_waste_model.pkl')
        print(f"\n💾 Best model saved: data/models/best_waste_model.pkl")
        
        # Save preprocessor for future predictions
        joblib.dump(preprocessor, 'data/models/preprocessor.pkl')
        print(f"💾 Preprocessor saved: data/models/preprocessor.pkl")
    
    # Step 8: Test Prediction on Sample Data
    print("\n7️⃣ SAMPLE WASTE PREDICTION")
    print("-"*40)
    
    # Make a sample prediction
    sample_idx = 0
    sample_features = X_test.iloc[[sample_idx]]
    actual_waste = y_test.iloc[sample_idx]
    
    predicted_waste = trainer.best_model.predict(sample_features)[0]
    
    print(f"\n📝 Sample Prediction:")
    print(f"   Actual waste: {actual_waste:.2f} kg")
    print(f"   Predicted waste: {predicted_waste:.2f} kg")
    print(f"   Error: {abs(actual_waste - predicted_waste):.2f} kg")
    print(f"   Accuracy: {100 - abs((actual_waste - predicted_waste) / actual_waste * 100):.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("✅ FOOD WASTE PREDICTION SYSTEM COMPLETE!")
    print("="*60)
    
    print(f"\n🏆 Best Model: {trainer.best_model_name}")
    print(f"📊 Average Prediction Error: {trainer.best_score:.2f} kg")
    print(f"📁 Plots saved in: data/plots/")
    print(f"📁 Models saved in: data/models/")
    
    # GPU Memory Summary
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    print("\n👋 System ready for food waste predictions!")

    # main_gpu.py (add this section at the end)

# ... (previous code remains the same) ...

    # Step 8: Future Prediction Module
    print("\n8️⃣ FUTURE PREDICTION MODULE")
    print("-"*40)
    
    # Ask if user wants to make future predictions
    choice = input("\n🔮 Would you like to predict future food waste? (y/n): ").strip().lower()
    
    if choice == 'y':
        from src.future_prediction import FutureWastePredictor
        
        # Initialize future predictor
        future_predictor = FutureWastePredictor(
            model_path='data/models/best_waste_model.pkl',
            preprocessor_path='data/models/preprocessor.pkl'
        )
        
        # Run interactive prediction
        future_predictor.run_interactive_prediction()
    
    print("\n" + "="*60)
    print("✅ FOOD WASTE PREDICTION SYSTEM COMPLETE!")
    print("="*60)
    
    print("\n👋 Thank you for using the Food Waste Prediction System!")

if __name__ == "__main__":
    main()