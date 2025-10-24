# main.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prediction import WastePredictor
from src.evaluation import ModelEvaluator

def main():
    """Main pipeline for Food Waste Prediction System"""
    
    print("\n" + "="*60)
    print("🍔 FOOD WASTE PREDICTION SYSTEM")
    print("="*60)
    
    # Step 1: Generate sample data (if needed)
    print("\n1️⃣ DATA GENERATION")
    print("-"*40)
    
    import os
    if not os.path.exists('data/raw/food_waste_data.csv'):
        print("📊 Generating sample data...")
        from generate_sample_data import generate_sample_data
        generate_sample_data(365)
    else:
        print("✅ Data file found")
    
    # Step 2: Data Preprocessing
    print("\n2️⃣ DATA PREPROCESSING")
    print("-"*40)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, processed_df = preprocessor.prepare_data(
        'data/raw/food_waste_data.csv'
    )
    
    # Step 3: Model Training
    print("\n3️⃣ MODEL TRAINING")
    print("-"*40)
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, perform_tuning=True)
    
    # Save models
    saved_models = trainer.save_all_models(results)
    
    # Step 4: Model Evaluation
    print("\n4️⃣ MODEL EVALUATION")
    print("-"*40)
    
    evaluator = ModelEvaluator()
    comparison_df = evaluator.evaluate_multiple_models(results, X_test, y_test)
    
    # Plot results
    print("\n📊 Generating evaluation plots...")
    evaluator.plot_predictions(save_path='data/models/predictions_plot.png')
    evaluator.plot_residuals(model_name=trainer.best_model_name, 
                            save_path='data/models/residuals_plot.png')
    
    # Feature importance for best model
    if hasattr(trainer.best_model, 'feature_importances_'):
        importance_df = evaluator.plot_feature_importance(
            trainer.best_model,
            preprocessor.feature_columns,
            trainer.best_model_name,
            save_path='data/models/feature_importance.png'
        )
        
        if importance_df is not None:
            print("\n📊 Top 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
    
    # Step 5: Making Predictions
    print("\n5️⃣ MAKING PREDICTIONS")
    print("-"*40)
    
    predictor = WastePredictor('data/models/best_model.pkl')
    predictor.load_preprocessor(preprocessor)
    
    # Example prediction
    sample_input = {
        'date': '2024-01-15',
        'food_item': 'Grilled Chicken',
        'food_category': 'mains',
        'quantity_prepared': 50,
        'quantity_sold': 35,
        'day_of_week': 'Monday',
        'is_weekend': 0,
        'weather': 'sunny',
        'temperature': 22,
        'special_event': 'none',
        'customer_count': 120,
        'month': 1,
        'unit_cost': 8.5
    }
    
    print("\n📝 Sample Prediction:")
    print(f"Input: {sample_input['food_item']} - {sample_input['quantity_prepared']} units prepared")
    
    result = predictor.predict_with_confidence(sample_input)
    print(f"Predicted waste: {result['predictions'][0]:.2f} units")
    print(f"Confidence interval: [{result['lower_bound'][0]:.2f}, {result['upper_bound'][0]:.2f}]")
    print(f"Estimated waste cost: ${result['predictions'][0] * sample_input['unit_cost']:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ SYSTEM READY FOR USE!")
    print("="*60)
    print(f"\n🏆 Best Model: {trainer.best_model_name}")
    print(f"📊 Test MAE: {trainer.best_score:.3f}")
    print(f"💾 Models saved in: data/models/")
    print(f"📈 Processed data saved in: data/processed/")
    
    # Interactive mode option
    print("\n" + "="*60)
    choice = input("\n🤔 Would you like to try real-time predictions? (y/n): ")
    if choice.lower() == 'y':
        predictor.real_time_prediction()
    
    print("\n👋 Thank you for using the Food Waste Prediction System!")

if __name__ == "__main__":
    main()